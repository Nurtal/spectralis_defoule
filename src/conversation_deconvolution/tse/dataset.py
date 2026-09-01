from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from conversation_deconvolution.synthetic.scenario import VOICES

if TYPE_CHECKING:
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
    def __init__(self, tts, config: TseConfig, dataset_dirs):
        if not dataset_dirs:
            raise ValueError("dataset_dirs must not be empty")
        self.tts = tts
        self.config = config
        self.dataset_dirs = [Path(d) for d in dataset_dirs]
        self._utterances: list[dict] = []
        self._load_utterances()
        if not self._utterances:
            raise ValueError(f"no utterances found in {dataset_dirs}")
        self._embedder = None
        speakers_sorted = sorted({u["speaker"] for u in self._utterances})
        self._speaker_to_voice = {
            spk: VOICES[i % len(VOICES)] for i, spk in enumerate(speakers_sorted)
        }

    def _load_utterances(self):
        for d in self.dataset_dirs:
            gt_path = d / "ground_truth.json"
            if not gt_path.exists():
                continue
            data = json.loads(gt_path.read_text())
            for conv in data["conversations"]:
                cid = str(conv.get("id", "conv"))
                for idx_u, u in enumerate(conv["utterances"]):
                    uid = str(u.get("id", f"{cid}_u{idx_u}"))
                    self._utterances.append(
                        {
                            "id": uid,
                            "speaker": u["speaker"],
                            "start": float(u["start"]),
                            "end": float(u["end"]),
                            "text": str(u.get("text", "")),
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
        voice_pool = rng.choice(VOICES, size=k, replace=False)
        voice_map = dict(zip(chosen_speakers, voice_pool))
        mix_utts: dict[str, dict] = {}
        for spk in chosen_speakers:
            pool = [u for u in self._utterances if u["speaker"] == spk]
            mix_utts[spk] = rng.choice(pool)
        target_idx = int(rng.integers(k))
        target_speaker = str(chosen_speakers[target_idx])
        mix_utt_for_target = mix_utts[target_speaker]
        pool = [
            u
            for u in self._utterances
            if u["speaker"] == target_speaker and u["id"] != mix_utt_for_target["id"]
        ]
        if pool:
            ref_utt = rng.choice(pool)
        else:
            ref_utt = mix_utt_for_target

        sr = 16000
        dur = 1.0
        mix = np.zeros(int(dur * sr), dtype=np.float32)
        target = np.zeros(int(dur * sr), dtype=np.float32)

        for spk in chosen_speakers:
            mix_utt = mix_utts[spk]
            voice = voice_map.get(spk, VOICES[0])
            audio, _ = self.tts.synthesize(mix_utt["text"], voice)
            audio = np.asarray(audio, dtype=np.float32)
            seg_dur = float(mix_utt["end"] - mix_utt["start"])
            start_s = float(rng.uniform(0, max(0, dur - seg_dur)))
            s = int(start_s * sr)
            e = min(s + int(seg_dur * sr), len(mix))
            seg_len = e - s
            take = min(seg_len, len(audio))
            if take > 0:
                mix[s : s + take] += audio[:take]
                if spk == target_speaker:
                    target[s : s + take] += audio[:take]

        ref_voice = voice_map.get(ref_utt["speaker"], VOICES[0])
        ref_audio, _ = self.tts.synthesize(ref_utt["text"], ref_voice)
        ref_audio = np.asarray(ref_audio, dtype=np.float32)
        ref_emb = self._compute_embedding(ref_audio)

        noise = rng.standard_normal(len(mix)).astype(np.float32)
        noise = np.asarray(_bandlimit_noise(noise, self.config.noise_bandwidth, sr))
        noise = noise.astype(np.float32)
        signal_power = float(np.mean(target**2)) + 1e-8
        noise_power = float(np.mean(noise**2)) + 1e-8
        snr_low = float(self.config.snr_low)
        snr_high = float(getattr(self.config, "snr_high", snr_low))
        snr_db = float(rng.uniform(snr_low, snr_high)) if snr_high > snr_low else snr_low
        scale = np.sqrt(signal_power / noise_power * 10 ** (-snr_db / 10))
        noise = noise * scale
        mix = mix + noise

        return mix.astype(np.float32), target.astype(np.float32), ref_emb, int(k)

    @classmethod
    def from_existing(cls, tts, config, dataset_dir):
        return cls(tts, config, [dataset_dir])
