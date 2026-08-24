import numpy as np

from conversation_deconvolution.conversation.export import save_result
from conversation_deconvolution.core.config import PipelineConfig
from conversation_deconvolution.core.types import (
    SpeakerTurn,
    TranscriptResult,
    Utterance,
)


class DeconvolutionPipeline:
    def __init__(
        self,
        diarizer,
        separator,
        asr,
        reconstructor,
        config: PipelineConfig | None = None,
        stem_embedder=None,
    ):
        self.diarizer = diarizer
        self.separator = separator
        self.asr = asr
        self.reconstructor = reconstructor
        self.cfg = config or PipelineConfig()
        self.stem_embedder = stem_embedder

    def _exclusive_ref(self, turn, mix: np.ndarray, regions) -> np.ndarray:
        sr = 16000
        covered = [
            (max(r.segment.start, turn.start), min(r.segment.end, turn.end))
            for r in regions
        ]
        covered = [(s, e) for s, e in covered if e - s > 0]
        pieces = []
        cursor = turn.start
        for s, e in sorted(covered):
            if s > cursor:
                pieces.append((cursor, min(s, turn.end)))
            cursor = max(cursor, e)
        if cursor < turn.end:
            pieces.append((cursor, turn.end))
        total = sum(e - s for s, e in pieces)
        if total < 0.25:
            return mix[int(turn.start * sr) : int(turn.end * sr)]
        out = np.concatenate(
            [mix[int(s * sr) : int(e * sr)] for s, e in pieces if e > s] or [np.zeros(0, np.float32)]
        )
        return out

    def _embed(self, signals: list[np.ndarray]) -> list[np.ndarray]:
        if hasattr(self.stem_embedder, "encode"):
            return list(self.stem_embedder.encode(signals))
        return [self.stem_embedder.embed(s) for s in signals]

    def _enhance(self, chunk: np.ndarray, chunk_start_idx: int, turn, sep_result, cache) -> np.ndarray:
        sr = 16000
        if not sep_result.regions or self.stem_embedder is None:
            return chunk
        chunk_start = chunk_start_idx / sr
        chunk_end = chunk_start + len(chunk) / sr
        ref_emb = self._turn_reference(turn, sep_result, cache)
        if ref_emb is None:
            return chunk
        enhanced = chunk.copy()
        for region in sep_result.regions:
            rs = max(region.segment.start, chunk_start)
            re_ = min(region.segment.end, chunk_end)
            if re_ - rs <= 0.05:
                continue
            key = id(region)
            if key not in cache["stem_embs"]:
                cache["stem_embs"][key] = [self._unit(v) for v in self._embed(region.stems)]
            sims = [float(np.dot(ref_emb, se)) for se in cache["stem_embs"][key]]
            order = np.argsort(sims)[::-1]
            best = int(order[0])
            if sims[best] < self.cfg.separation.assign_min_sim:
                continue
            if len(sims) > 1 and sims[best] - float(sims[order[1]]) < self.cfg.separation.assign_min_margin:
                continue
            n = int((re_ - rs) * sr)
            stem = self._fit(region.stems[best], n)
            start_off = int(rs * sr) - chunk_start_idx
            enhanced[start_off : start_off + n] = stem
        return enhanced

    def _turn_reference(self, turn, sep_result, cache):
        centroids = getattr(self.diarizer, "speaker_centroids_", None)
        if centroids and turn.speaker:
            try:
                lab = int(turn.speaker.rsplit("_", 1)[-1])
            except ValueError:
                lab = None
            if lab is not None and lab in centroids:
                return centroids[lab]
        ref_key = (turn.speaker, turn.start, turn.end)
        if ref_key not in cache["refs"]:
            ref = self._exclusive_ref(turn, sep_result.mix, sep_result.regions)
            if len(ref) == 0:
                return None
            cache["refs"][ref_key] = self._unit(self._embed([ref])[0])
        return cache["refs"][ref_key]

    @staticmethod
    def _fit(signal: np.ndarray, n: int) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        if len(signal) >= n:
            return signal[:n]
        return np.pad(signal, (0, n - len(signal)))

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        norm = float(np.linalg.norm(v)) or 1.0
        return v / norm

    def run(self, audio: np.ndarray) -> TranscriptResult:
        from conversation_deconvolution.diarization.timeline import overlap_regions

        audio = np.asarray(audio, dtype=np.float32)
        turns, _embeddings = self.diarizer.diarize(audio)
        overlaps = getattr(self.diarizer, "overlap_regions_", None)
        if overlaps is None:
            overlaps = overlap_regions(turns)
        sep_result = self.separator.separate(audio, overlaps)
        mix = sep_result.mix

        utterances: list[Utterance] = []
        pad = self.cfg.asr.context_pad_sec
        cache: dict = {"refs": {}, "stem_embs": {}}
        for i, turn in enumerate(turns):
            s = max(0, int((turn.start - pad) * 16000))
            e = min(len(mix), int((turn.end + pad) * 16000))
            segment = self._enhance(mix[s:e], s, turn, sep_result, cache)
            asr_res = self.asr.transcribe(segment)
            utterances.append(
                Utterance(
                    id=f"utt_{i:03d}",
                    speaker=turn.speaker,
                    start=round(turn.start, 3),
                    end=round(turn.end, 3),
                    text=asr_res.text,
                    confidence=round(asr_res.confidence, 4),
                    language=asr_res.language,
                )
            )
        conversations = self.reconstructor.reconstruct(utterances)
        return TranscriptResult(
            utterances=utterances, conversations=conversations, overlaps=overlaps
        )

    def run_file(self, path, output_path) -> TranscriptResult:
        from conversation_deconvolution.audio.loader import load_audio

        result = self.run(load_audio(path))
        save_result(result, output_path)
        return result


def build_pipeline(config: PipelineConfig) -> DeconvolutionPipeline:
    from conversation_deconvolution.asr.faster_whisper_asr import FasterWhisperAsr
    from conversation_deconvolution.audio.vad import SileroVad
    from conversation_deconvolution.conversation.reconstructor import (
        HeuristicReconstructor,
    )
    from conversation_deconvolution.conversation.semantic import (
        SentenceTransformerEmbedder,
    )
    from conversation_deconvolution.diarization.clusterer import (
        AgglomerativeClusterer,
    )
    from conversation_deconvolution.diarization.diarizer import SpeakerDiarizer
    from conversation_deconvolution.diarization.embeddings import EcapaEmbedder
    from conversation_deconvolution.separation.passthrough import (
        PassthroughSeparator,
    )
    from conversation_deconvolution.separation.sepformer import SepformerSeparator

    vad = SileroVad(config.vad)
    embedder = EcapaEmbedder()
    clusterer = AgglomerativeClusterer(config.diarization.distance_threshold)
    diarizer = SpeakerDiarizer(vad, embedder, clusterer, config.diarization)
    text_embedder = SentenceTransformerEmbedder(config.text_embedding_model)
    reconstructor = HeuristicReconstructor(text_embedder, config.reconstruction)
    separator = (
        SepformerSeparator(config.separation)
        if config.separation.enabled
        else PassthroughSeparator()
    )
    return DeconvolutionPipeline(
        diarizer=diarizer,
        separator=separator,
        asr=FasterWhisperAsr(config.asr),
        reconstructor=reconstructor,
        config=config,
        stem_embedder=embedder,
    )


def turns_from_conversations(conversations) -> list[SpeakerTurn]:
    turns = [
        SpeakerTurn(u.speaker or "?", u.start, u.end)
        for c in conversations
        for u in c.utterances
    ]
    return sorted(turns, key=lambda t: t.start)
