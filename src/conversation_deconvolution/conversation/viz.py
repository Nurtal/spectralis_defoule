import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from conversation_deconvolution.core.types import TranscriptResult


def plot_timeline(result: TranscriptResult, out_png, title: str = "Timeline") -> str:
    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.6 * _n_speakers(result)))
    speakers = sorted({u.speaker for u in result.utterances if u.speaker})
    conv_of: dict[float, int] = {}
    for ci, conv in enumerate(result.conversations):
        for u in conv.utterances:
            conv_of[(u.start, u.end)] = ci
    palette = plt.get_cmap("tab10")
    for row, spk in enumerate(speakers):
        for u in result.utterances:
            if u.speaker != spk:
                continue
            color = palette(conv_of.get((u.start, u.end), 9) % 10)
            ax.broken_barh(
                [(u.start, u.duration)],
                (row - 0.35, 0.7),
                facecolors=color,
                edgecolor="black",
                linewidth=0.4,
            )
    for seg in result.overlaps:
        ax.axvspan(seg.start, seg.end, color="red", alpha=0.12, zorder=0)
    ax.set_yticks(range(len(speakers)), labels=speakers)
    ax.set_xlabel("temps (s)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return str(out_png)


def _n_speakers(result: TranscriptResult) -> int:
    return max(1, len({u.speaker for u in result.utterances if u.speaker}))
