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
            (max(r.segment.start, turn.start), min(r.segment.end, turn.end)) for r in regions
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
            [mix[int(s * sr) : int(e * sr)] for s, e in pieces if e > s]
            or [np.zeros(0, np.float32)]
        )
        return out

    def _embed(self, signals: list[np.ndarray]) -> list[np.ndarray]:
        if hasattr(self.stem_embedder, "encode"):
            return list(self.stem_embedder.encode(signals))
        return [self.stem_embedder.embed(s) for s in signals]

    def _assign_best_stem(self, turn, region, ref_emb, cache) -> np.ndarray | None:
        sr = 16000
        rs = max(region.segment.start, turn.start)
        re_ = min(region.segment.end, turn.end)
        if re_ - rs <= 0.05:
            return None
        key = id(region)
        if key not in cache["stem_embs"]:
            cache["stem_embs"][key] = [self._unit(v) for v in self._embed(region.stems)]
        sims = [float(np.dot(ref_emb, se)) for se in cache["stem_embs"][key]]
        order = np.argsort(sims)[::-1]
        top = int(order[0])
        if sims[top] < self.cfg.separation.assign_min_sim:
            return None
        if (
            len(sims) > 1
            and sims[top] - float(sims[order[1]]) < self.cfg.separation.assign_min_margin
        ):
            return None
        reg_off = int(region.segment.start * sr)
        seg_s = int(rs * sr) - reg_off
        seg_e = int(re_ * sr) - reg_off
        return np.asarray(region.stems[top][seg_s:seg_e], dtype=np.float32)

    def _turn_segments(self, turn, sep_result, cache) -> list[tuple[float, float, np.ndarray]]:
        sr = 16000
        mix = sep_result.mix

        if not sep_result.regions or self.stem_embedder is None:
            s = int(turn.start * sr)
            e = min(len(mix), int(turn.end * sr))
            return [(turn.start, turn.end, mix[s:e])]

        ref_emb = self._turn_reference(turn, sep_result, cache)
        if ref_emb is None:
            s = int(turn.start * sr)
            e = min(len(mix), int(turn.end * sr))
            return [(turn.start, turn.end, mix[s:e])]

        stem_parts: list[tuple[float, float, np.ndarray]] = []
        for region in sep_result.regions:
            stem_audio = self._assign_best_stem(turn, region, ref_emb, cache)
            if stem_audio is not None:
                rs = max(region.segment.start, turn.start)
                re_ = min(region.segment.end, turn.end)
                stem_parts.append((rs, re_, stem_audio))

        stem_parts.sort(key=lambda t: t[0])

        segments: list[tuple[float, float, np.ndarray]] = []
        cursor = turn.start
        for rs, re_, stem_audio in stem_parts:
            if rs > cursor + 0.01:
                s = int(cursor * sr)
                e = min(len(mix), int(rs * sr))
                if e > s:
                    segments.append((cursor, rs, mix[s:e]))
            segments.append((rs, re_, stem_audio))
            cursor = re_
        if cursor < turn.end - 0.01:
            s = int(cursor * sr)
            e = min(len(mix), int(turn.end * sr))
            if e > s:
                segments.append((cursor, turn.end, mix[s:e]))

        if not segments:
            s = int(turn.start * sr)
            e = min(len(mix), int(turn.end * sr))
            segments = [(turn.start, turn.end, mix[s:e])]
        return segments

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
            vec = np.asarray(self._embed([ref])[0], dtype=np.float64)
            if len(ref) == 0 or float(np.linalg.norm(vec)) < 1e-8:
                return None
            cache["refs"][ref_key] = self._unit(vec)
        return cache["refs"][ref_key]

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

        utterances: list[Utterance] = []
        cache: dict = {"refs": {}, "stem_embs": {}}
        for i, turn in enumerate(turns):
            segments = self._turn_segments(turn, sep_result, cache)
            parts = []
            total_conf = 0.0
            for _, _, seg_audio in segments:
                if len(seg_audio) == 0:
                    continue
                asr_res = self.asr.transcribe(seg_audio)
                parts.append(asr_res.text)
                total_conf += asr_res.confidence
            text = " ".join(parts)
            conf = total_conf / max(len(parts), 1)
            utterances.append(
                Utterance(
                    id=f"utt_{i:03d}",
                    speaker=turn.speaker,
                    start=round(turn.start, 3),
                    end=round(turn.end, 3),
                    text=text,
                    confidence=round(conf, 4),
                    language="fr",
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


def build_reconstructor(config: PipelineConfig, text_embedder):
    if config.reconstructor_kind == "graph":
        from conversation_deconvolution.conversation.graph_reconstructor import (
            GraphReconstructor,
        )

        return GraphReconstructor(text_embedder, config.graph)
    if config.reconstructor_kind == "heuristic":
        from conversation_deconvolution.conversation.reconstructor import (
            HeuristicReconstructor,
        )

        return HeuristicReconstructor(text_embedder, config.reconstruction)
    raise ValueError(f"unknown reconstructor_kind: {config.reconstructor_kind}")


def build_pipeline(config: PipelineConfig) -> DeconvolutionPipeline:
    from conversation_deconvolution.asr.faster_whisper_asr import FasterWhisperAsr
    from conversation_deconvolution.audio.vad import SileroVad
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
    reconstructor = build_reconstructor(config, text_embedder)
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
