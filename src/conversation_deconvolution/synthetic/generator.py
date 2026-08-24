from pathlib import Path

import numpy as np
import soundfile as sf

from conversation_deconvolution.conversation.export import save_json
from conversation_deconvolution.core.config import SyntheticConfig
from conversation_deconvolution.core.types import Conversation, Utterance, conversation_to_dict
from conversation_deconvolution.synthetic.mixer import add_noise, place
from conversation_deconvolution.synthetic.scenario import generate_scenario


class SyntheticGenerator:
    def __init__(self, tts, config: SyntheticConfig):
        self.tts = tts
        self.cfg = config

    def generate(
        self,
        out_dir: str | Path,
        seed: int,
        n_conversations: int = 2,
        speakers_per_thread: int = 2,
        n_lines: tuple[int, int] = (4, 8),
    ) -> Path:
        rng = np.random.default_rng(seed)
        threads = generate_scenario(
            n_conversations=n_conversations,
            speakers_per_thread=speakers_per_thread,
            n_lines=n_lines,
            rng=rng,
            mean_gap_sec=self.cfg.mean_gap_sec,
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
        noisy = add_noise(mix, self.cfg.snr_db, rng)

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sf.write(out_dir / "mixed.wav", noisy, self.cfg.sample_rate)
        save_json(
            out_dir / "ground_truth.json",
            {"conversations": [conversation_to_dict(c) for c in conversations]},
        )
        return out_dir
