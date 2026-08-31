import json
import numpy as np
from pathlib import Path

from conversation_deconvolution.synthetic.tts import PiperTts
from conversation_deconvolution.core.config import TseConfig


def _bandlimit_noise(noise, bandwidth_hz, sr=16000):
    if bandwidth_hz <= 0:
        return noise
    from scipy.signal import butter, filtfilt
    nyq = sr / 2
    low = bandwidth_hz / nyq
    b, a = butter(4, low, btype="low")
    return filtfilt(b, a, noise)


class TseDataset:
    def __init__(self, tts, config, dataset_dirs):
        self.tts = tts
        self.config = config
        self.dataset_dirs = [Path(d) for d in dataset_dirs]
        self._utterances = []
        self._load_utterances()
        self._embedder = None

    def _load_utterances(self):
        for d in self.dataset_dirs:
            gt_path = d / "ground_truth.json"
            if not gt_path.exists():
                continue
            data = json.loads(gt_path.read_text())
            for conv in data["conversations"]:
                for u in conv["utterances"]:
                    self._utterances.append(
                        {
                            "id": u["id"],
                            "speaker": u["speaker"],
                            "start": u["start"],
                            "end": u["end"],
                            "text": u["text"],
                        }
                    )

    def _get_embedder(self):
        if self._embedder is None:
            from conversation_deconvolution.diarization.embeddings import EcapaEmbedder
            self._embedder = EcapaEmbedder()
        return self._embedder

    def _compute_embedding(self, audio):
        embedder = self._get_embedder()
        emb = np.asarray(embedder.embed(audio), dtype=np.float64)
        norm = float(np.linalg.norm(emb)) or 1.0
        return emb / norm

    def __len__(self):
        return max(1, len(self._utterances) // max(1, self.config.batch_size))

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        speakers = sorted({u["speaker"] for u in self._utterances})
        k = rng.integers(2, min(5, len(speakers)) + 1)
        chosen_speakers = rng.choice(speakers, size=k, replace=False)
        selected = [u for u in self._utterances if u["speaker"] in chosen_speakers]
        rng.shuffle(selected)
        selected = selected[:k]

        sr = 16000
        dur = 1.0
        mix = np.zeros(int(dur * sr), dtype=np.float32)
        target = np.zeros(int(dur * sr), dtype=np.float32)
        target_idx = rng.integers(k)
        target_speaker = chosen_speakers[target_idx]
        ref_utt = None

        for i, spk in enumerate(chosen_speakers):
            spk_utt = [u for u in selected if u["speaker"] == spk]
            if not spk_utt:
                continue
            if spk == target_speaker and len(spk_utt) > 1:
                mix_utt, ref_utt = spk_utt[0], spk_utt[1]
            elif spk == target_speaker:
                mix_utt, ref_utt = spk_utt[0], spk_utt[0]
            else:
                mix_utt, ref_utt = spk_utt[0], None

            audio, _ = self.tts.synthesize(mix_utt["text"], mix_utt["speaker"])
            audio = np.asarray(audio, dtype=np.float32)
            seg_dur = mix_utt["end"] - mix_utt["start"]
            start_s = float(rng.uniform(0, max(0, dur - seg_dur)))
            s = int(start_s * sr)
            e = min(s + int(seg_dur * sr), len(mix))
            seg_len = e - s
            if seg_len > 0 and len(audio) >= seg_len:
                mix[s:e] += audio[:seg_len]
                if spk == target_speaker:
                    target[s:e] += audio[:seg_len]

        if ref_utt is None:
            ref_utt = selected[0]
        ref_audio, _ = self.tts.synthesize(ref_utt["text"], ref_utt["speaker"])
        ref_audio = np.asarray(ref_audio, dtype=np.float32)
        ref_emb = self._compute_embedding(ref_audio)

        noise = rng.randn(len(mix)).astype(np.float32)
        noise = _bandlimit_noise(noise, self.config.noise_bandwidth, sr)
        signal_power = float(np.mean(target ** 2)) + 1e-8
        noise_power = float(np.mean(noise ** 2)) + 1e-8
        scale = np.sqrt(signal_power / noise_power * 10 ** (-self.config.snr_low / 10))
        noise = noise * scale
        mix = mix + noise

        return mix.astype(np.float32), target.astype(np.float32), ref_emb, len(chosen_speakers)

    @classmethod
    def from_existing(cls, tts, config, dataset_dir):
        return cls(tts, config, [dataset_dir])