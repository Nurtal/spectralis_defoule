import math

import numpy as np

from conversation_deconvolution.core.types import Segment, SpeakerTurn


def make_windows(
    segments: list[Segment], window_sec: float, hop_sec: float, min_window_sec: float = 0.25
) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for seg in segments:
        if seg.duration <= window_sec + 1e-9:
            windows.append((seg.start, seg.end))
            continue
        local: list[tuple[float, float]] = []
        t = seg.start
        while t < seg.end - 1e-9:
            end = min(t + window_sec, seg.end)
            local.append((t, end))
            t += hop_sec
        if local and local[-1][1] - local[-1][0] < min_window_sec and len(local) > 1:
            prev_start = local[-2][0]
            local[-2] = (prev_start, seg.end)
            local.pop()
        windows.extend(local)
    return windows


def _vote_cells(
    windows: list[tuple[float, float]],
    labels: list[int],
    t0: float,
    n_cells: int,
    cell_sec: float,
) -> np.ndarray:
    n_labels = max(labels) + 1
    votes = np.zeros((n_cells, n_labels))
    for (ws, we), lab in zip(windows, labels):
        c0 = int((ws - t0) / cell_sec)
        c1 = math.ceil((we - t0) / cell_sec)
        for c in range(max(0, c0), min(n_cells, c1)):
            cs = c * cell_sec
            overlap = min(we, cs + cell_sec) - max(ws, cs)
            if overlap > 0:
                votes[c, lab] += overlap
    return votes


def windows_to_turns(
    windows: list[tuple[float, float]],
    labels: list[int],
    cell_sec: float = 0.25,
    min_turn_sec: float = 0.3,
) -> list[tuple[int, float, float]]:
    if not windows:
        return []
    t0 = min(s for s, _ in windows)
    t1 = max(e for _, e in windows)
    n_cells = max(1, math.ceil((t1 - t0 - 1e-9) / cell_sec))
    votes = _vote_cells(windows, labels, t0, n_cells, cell_sec)

    assigned: list[int] = []
    prev = -1
    for c in range(n_cells):
        col = votes[c]
        if col.sum() <= 0:
            best = max(prev, 0)
        else:
            best = int(np.argmax(col))
            if prev >= 0 and col[prev] == col[best]:
                best = prev
        assigned.append(best)
        prev = best

    runs: list[tuple[int, float, float]] = []
    for c, lab in enumerate(assigned):
        start = c * cell_sec
        end = start + cell_sec
        if runs and runs[-1][0] == lab:
            runs[-1] = (lab, runs[-1][1], end)
        else:
            runs.append((lab, start, end))

    absorbed: list[tuple[int, float, float]] = []
    for idx, (lab, start, end) in enumerate(runs):
        if end - start < min_turn_sec and absorbed:
            plab, pstart, _pend = absorbed[-1]
            absorbed[-1] = (plab, pstart, end)
            continue
        absorbed.append((lab, start, end))
    if len(absorbed) > 1 and absorbed[-1][2] - absorbed[-1][1] < min_turn_sec:
        plab, pstart, _pend = absorbed[-2]
        absorbed[-2] = (plab, pstart, absorbed[-1][2])
        absorbed.pop()

    return [(lab, round(t0 + start, 4), round(t0 + end, 4)) for lab, start, end in absorbed]


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


def margin_regions(
    windows: list[tuple[float, float]],
    labels: list[int],
    cell_sec: float = 0.25,
    min_margin: float = 0.34,
    min_duration: float = 0.3,
) -> list[Segment]:
    if not windows:
        return []
    t0 = min(s for s, _ in windows)
    t1 = max(e for _, e in windows)
    n_cells = max(1, math.ceil((t1 - t0 - 1e-9) / cell_sec))
    votes = _vote_cells(windows, labels, t0, n_cells, cell_sec)
    contested: list[float] = []
    for c in range(n_cells):
        col = votes[c]
        total = float(col.sum())
        if total <= 0:
            contested.append(1.0)
            continue
        ordered = np.sort(col)[::-1]
        top2 = float(ordered[1]) if len(ordered) > 1 else 0.0
        contested.append((float(ordered[0]) - top2) / total)

    regions = []
    start = None
    for c in range(n_cells + 1):
        low = c < n_cells and contested[c] < min_margin
        if low and start is None:
            start = c
        elif not low and start is not None:
            seg_start = t0 + start * cell_sec
            seg_end = t0 + c * cell_sec
            if seg_end - seg_start >= min_duration:
                regions.append(Segment(seg_start, seg_end))
            start = None
    return regions


def merge_segments(segments: list[Segment], gap: float = 0.1) -> list[Segment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    merged = [ordered[0]]
    for seg in ordered[1:]:
        last = merged[-1]
        if seg.start <= last.end + gap:
            if seg.end > last.end:
                merged[-1] = Segment(last.start, seg.end)
        else:
            merged.append(seg)
    return merged
