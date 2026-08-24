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
    ):
        self.diarizer = diarizer
        self.separator = separator
        self.asr = asr
        self.reconstructor = reconstructor
        self.cfg = config or PipelineConfig()

    def run(self, audio: np.ndarray) -> TranscriptResult:
        from conversation_deconvolution.diarization.timeline import overlap_regions

        audio = np.asarray(audio, dtype=np.float32)
        turns, _embeddings = self.diarizer.diarize(audio)
        overlaps = overlap_regions(turns)
        sources = self.separator.separate(audio, overlaps)
        mix = sources[0]

        utterances: list[Utterance] = []
        pad = self.cfg.asr.context_pad_sec
        for i, turn in enumerate(turns):
            s = max(0, int((turn.start - pad) * 16000))
            e = min(len(mix), int((turn.end + pad) * 16000))
            asr_res = self.asr.transcribe(mix[s:e])
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

    vad = SileroVad(config.vad)
    embedder = EcapaEmbedder()
    clusterer = AgglomerativeClusterer(config.diarization.distance_threshold)
    diarizer = SpeakerDiarizer(vad, embedder, clusterer, config.diarization)
    text_embedder = SentenceTransformerEmbedder(config.text_embedding_model)
    reconstructor = HeuristicReconstructor(text_embedder, config.reconstruction)
    return DeconvolutionPipeline(
        diarizer=diarizer,
        separator=PassthroughSeparator(),
        asr=FasterWhisperAsr(config.asr),
        reconstructor=reconstructor,
        config=config,
    )


def turns_from_conversations(conversations) -> list[SpeakerTurn]:
    turns = [
        SpeakerTurn(u.speaker or "?", u.start, u.end)
        for c in conversations
        for u in c.utterances
    ]
    return sorted(turns, key=lambda t: t.start)
