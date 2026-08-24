from conversation_deconvolution.core.types import Segment, SpeakerTurn


def merge_turns(turns: list[SpeakerTurn], gap: float = 0.2) -> list[SpeakerTurn]:
    ordered = sorted(turns, key=lambda t: (t.start, t.end))
    merged: list[SpeakerTurn] = []
    for t in ordered:
        if merged and merged[-1].speaker == t.speaker and t.start - merged[-1].end <= gap:
            last = merged[-1]
            merged[-1] = SpeakerTurn(last.speaker, last.start, max(last.end, t.end))
        else:
            merged.append(t)
    return merged


def overlap_regions(turns: list[SpeakerTurn], min_duration: float = 0.05) -> list[Segment]:
    events: list[tuple[float, int, str]] = []
    for t in turns:
        events.append((t.start, 1, t.speaker))
        events.append((t.end, -1, t.speaker))
    events.sort(key=lambda e: (e[0], e[1]))
    active: dict[str, int] = {}
    overlaps: list[Segment] = []
    prev_t: float | None = None
    for time, kind, speaker in events:
        if prev_t is not None and time > prev_t and len(active) >= 2:
            seg = Segment(prev_t, time)
            if seg.duration >= min_duration:
                overlaps.append(seg)
        if kind == 1:
            active[speaker] = active.get(speaker, 0) + 1
        else:
            active[speaker] -= 1
            if active[speaker] <= 0:
                del active[speaker]
        prev_t = time
    return merge_adjacent(overlaps)


def merge_adjacent(segments: list[Segment], eps: float = 1e-9) -> list[Segment]:
    if not segments:
        return []
    out = [segments[0]]
    for s in segments[1:]:
        last = out[-1]
        if s.start - last.end <= eps:
            out[-1] = Segment(last.start, max(last.end, s.end))
        else:
            out.append(s)
    return out
