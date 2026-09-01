from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from conversation_deconvolution.conversation.export import save_json
from conversation_deconvolution.core.config import SyntheticConfig
from conversation_deconvolution.core.types import Conversation, Utterance, conversation_to_dict
from conversation_deconvolution.synthetic.mixer import add_noise, place
from conversation_deconvolution.synthetic.scenario import generate_scenario


class SyntheticGenerator:
    def __init__(self, tts, config: SyntheticConfig):
        self.tts = tts
        self.cfg = config

    def _apply_gain(self, audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.cfg.gain_low_db == 0 and self.cfg.gain_high_db == 0:
            return audio
        gain_db = float(rng.uniform(self.cfg.gain_low_db, self.cfg.gain_high_db))
        return (audio * (10 ** (gain_db / 20.0))).astype(np.float32)

    def _apply_rir(self, audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if rng.random() >= self.cfg.rir_prob:
            return audio
        sr = self.cfg.sample_rate
        rir_len = int(sr * 0.25)
        t = np.arange(rir_len) / sr
        rir = np.exp(-t * self.cfg.rir_decay * 10)
        rir[0] = 1.0
        rir /= np.sqrt(np.sum(rir**2)) + 1e-8
        out = fftconvolve(audio.astype(np.float64), rir.astype(np.float64), mode="full")
        out = out[: len(audio)].astype(np.float32)
        peak = float(np.max(np.abs(out))) or 1.0
        if peak > 0.95:
            out = (out / peak * 0.9).astype(np.float32)
        return out

    def generate(
        self,
        out_dir: str | Path,
        seed: int,
        n_conversations: int = 2,
        speakers_per_thread: int = 2,
        n_lines: tuple[int, int] = (4, 8),
    ) -> Path:
        rng = np.random.default_rng(seed)
        mean_gap = self.cfg.mean_gap_sec
        if (
            hasattr(self.cfg, "mean_gap_low")
            and hasattr(self.cfg, "mean_gap_high")
            and self.cfg.mean_gap_low != self.cfg.mean_gap_high
        ):
            mean_gap = float(rng.uniform(self.cfg.mean_gap_low, self.cfg.mean_gap_high))
        threads = generate_scenario(
            n_conversations=n_conversations,
            speakers_per_thread=speakers_per_thread,
            n_lines=n_lines,
            rng=rng,
            mean_gap_sec=mean_gap,
        )
        clips: list[tuple[float, np.ndarray]] = []
        conversations: list[Conversation] = []
        for conv_idx, thread in enumerate(threads):
            offset = float(rng.uniform(0.0, 1.5)) * conv_idx
            cursor = offset
            utterances = []
            gaps = [0.0] + list(thread.gaps)
            for k, (line, gap) in enumerate(zip(thread.lines, gaps)):
                audio, sr = self.tts.synthesize(line.text, line.voice)
                audio = self._apply_gain(np.asarray(audio, dtype=np.float32), rng)
                audio = self._apply_rir(audio, rng)
                duration = len(audio) / sr
                uid = f"conversation_{conv_idx + 1:02d}_u{k:02d}"
                utterances.append(
                    Utterance(
                        id=uid,
                        speaker=line.speaker,
                        start=round(cursor, 3),
                        end=round(cursor + duration, 3),
                        text=line.text,
                    )
                )
                clips.append((cursor, audio))
                cursor += duration + gap
            conversations.append(
                Conversation(
                    id=f"conversation_{conv_idx + 1:02d}",
                    participants=list(thread.speakers),
                    utterances=utterances,
                )
            )
        total_end = max(u.end for c in conversations for u in c.utterances) + 1.0
        mix = place(clips, int(total_end * self.cfg.sample_rate))
        snr_db = self.cfg.snr_db
        if (
            hasattr(self.cfg, "snr_low")
            and hasattr(self.cfg, "snr_high")
            and self.cfg.snr_low != self.cfg.snr_high
        ):
            snr_db = float(rng.uniform(self.cfg.snr_low, self.cfg.snr_high))
        noisy = add_noise(mix, snr_db, rng)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / "mixed.wav", noisy, self.cfg.sample_rate)
        save_json(
            out_dir / "ground_truth.json",
            {"conversations": [conversation_to_dict(c) for c in conversations]},
        )
        return out_dir
