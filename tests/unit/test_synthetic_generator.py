import conversation_deconvolution.conversation.export as export_mod
from conversation_deconvolution.core.config import SyntheticConfig
from conversation_deconvolution.core.types import conversation_from_dict
from conversation_deconvolution.synthetic.generator import SyntheticGenerator


class FakeTts:
    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice):
        self.calls.append((text, voice))
        dur = 0.3 + 0.01 * len(text.split())
        n = int(dur * 16000)
        t = __import__("numpy").arange(n) / 16000
        freq = 200 + hash(voice) % 300
        return (0.4 * __import__("numpy").sin(2 * 3.14159 * freq * t)).astype("float32"), 16000


def test_generator_produces_dataset(tmp_path):
    tts = FakeTts()
    gen = SyntheticGenerator(tts, SyntheticConfig(sample_rate=16000, snr_db=12.0, mean_gap_sec=0.5))
    out = gen.generate(tmp_path / "ds", seed=11, n_conversations=2, speakers_per_thread=2, n_lines=(3, 5))
    assert (out / "mixed.wav").exists()
    gt = export_mod.load_json(out / "ground_truth.json")
    assert len(gt["conversations"]) == 2
    convs = [conversation_from_dict(c) for c in gt["conversations"]]
    all_utts = [u for c in convs for u in c.utterances]
    assert len(all_utts) >= 6
    assert all(u.end > u.start for u in all_utts)
    participants = {p for c in convs for p in c.participants}
    assert len(participants) == 4
    data, sr = __import__("soundfile").read(out / "mixed.wav")
    assert sr == 16000 and abs(len(data) / sr - max(u.end for u in all_utts)) < 1.5


def test_generator_seed_reproducible(tmp_path):
    gen = lambda: SyntheticGenerator(FakeTts(), SyntheticConfig(mean_gap_sec=0.5))
    a = gen().generate(tmp_path / "a", seed=3, n_lines=(3, 3))
    b = gen().generate(tmp_path / "b", seed=3, n_lines=(3, 3))
    ta = (a / "ground_truth.json").read_text()
    tb = (b / "ground_truth.json").read_text()
    assert ta == tb
