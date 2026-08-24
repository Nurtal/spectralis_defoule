from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class VadResult:
    segments: list[Segment]
    frame_probs: object
    frame_rate: float


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Utterance:
    id: str
    speaker: str | None
    start: float
    end: float
    text: str = ""
    confidence: float | None = None
    language: str | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Conversation:
    id: str
    participants: list[str] = field(default_factory=list)
    utterances: list[Utterance] = field(default_factory=list)


@dataclass
class TranscriptResult:
    utterances: list[Utterance] = field(default_factory=list)
    conversations: list[Conversation] = field(default_factory=list)
    overlaps: list[Segment] = field(default_factory=list)


def utterance_to_dict(u: Utterance) -> dict:
    d = {
        "speaker": u.speaker,
        "start": u.start,
        "end": u.end,
        "text": u.text,
    }
    if u.confidence is not None:
        d["confidence"] = u.confidence
    if u.language is not None:
        d["language"] = u.language
    return d


def utterance_from_dict(d: dict, id_: str | None = None) -> Utterance:
    return Utterance(
        id=id_ or d.get("id", ""),
        speaker=d.get("speaker"),
        start=float(d["start"]),
        end=float(d["end"]),
        text=d.get("text", ""),
        confidence=d.get("confidence"),
        language=d.get("language"),
    )


def conversation_to_dict(c: Conversation) -> dict:
    return {
        "id": c.id,
        "participants": list(c.participants),
        "utterances": [utterance_to_dict(u) for u in c.utterances],
    }


def conversation_from_dict(d: dict) -> Conversation:
    return Conversation(
        id=d["id"],
        participants=list(d.get("participants", [])),
        utterances=[
            utterance_from_dict(u, f"{d['id']}_{i}")
            for i, u in enumerate(d.get("utterances", []))
        ],
    )


def result_to_dict(r: TranscriptResult) -> dict:
    return {"conversations": [conversation_to_dict(c) for c in r.conversations]}


def result_from_dict(d: dict) -> TranscriptResult:
    convs = [conversation_from_dict(c) for c in d.get("conversations", [])]
    return conversations_to_result(convs)


def conversations_to_result(conversations: list[Conversation]) -> TranscriptResult:
    utterances = [u for c in conversations for u in c.utterances]
    utterances.sort(key=lambda u: (u.start, u.end))
    return TranscriptResult(utterances=utterances, conversations=conversations)


def asdict_shallow(obj) -> dict:
    return asdict(obj)
