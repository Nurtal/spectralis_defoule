from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from conversation_deconvolution.core.config import PipelineConfig
from conversation_deconvolution.evaluation.clustering_metrics import (
    conversation_metrics,
)
from conversation_deconvolution.evaluation.der import diarization_error_rate
from conversation_deconvolution.evaluation.wer import match_by_iou, wer_report

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def run(
    input: Path = typer.Argument(..., exists=True),
    output: Path = typer.Option("out.json", "--output", "-o"),
    config_path: Path = typer.Option(None, "--config", "-c"),
    num_speakers: int = typer.Option(None, "--num-speakers", "-n"),
    plot: Path = typer.Option(None, "--plot", "-p"),
    separate: bool = typer.Option(None, "--separate/--no-separate"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    if num_speakers:
        cfg.diarization.num_speakers = num_speakers
    if separate is not None:
        cfg.separation.enabled = separate
    from conversation_deconvolution.conversation.viz import plot_timeline
    from conversation_deconvolution.pipeline import build_pipeline

    pipeline = build_pipeline(cfg)
    with console.status("[bold green]Traitement audio…"):
        result = pipeline.run_file(input, output)
    console.print(
        f"[green]✓[/green] {len(result.utterances)} énoncés, "
        f"{len(result.conversations)} conversations → {output}"
    )
    if plot:
        plot_timeline(result, plot, title=Path(input).name)
        console.print(f"[green]✓[/green] timeline → {plot}")


@app.command()
def synth(
    out_dir: Path = typer.Option("data/synthetic/sample", "--out", "-o"),
    conversations: int = typer.Option(2, "--conversations"),
    speakers: int = typer.Option(2, "--speakers"),
    seed: int = typer.Option(0, "--seed"),
    snr_db: float = typer.Option(None, "--snr-db"),
    config_path: Path = typer.Option(None, "--config", "-c"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    if snr_db is not None:
        cfg.synthetic.snr_db = snr_db
    from conversation_deconvolution.synthetic.generator import SyntheticGenerator
    from conversation_deconvolution.synthetic.tts import PiperTts

    gen = SyntheticGenerator(PiperTts(), cfg.synthetic)
    path = gen.generate(
        out_dir, seed=seed, n_conversations=conversations, speakers_per_thread=speakers
    )
    console.print(f"[green]✓[/green] dataset → {path} (mixed.wav + ground_truth.json)")


@app.command()
def evaluate(
    pred: Path = typer.Option(..., "--pred", "-p", exists=True),
    gt: Path = typer.Option(..., "--gt", "-g", exists=True),
):
    from conversation_deconvolution.core.types import result_from_dict

    hyp = result_from_dict(_load(pred))
    ref = result_from_dict(_load(gt))
    metrics = evaluate_results(ref, hyp)
    table = Table(title="Évaluation")
    table.add_column("Métrique")
    table.add_column("Valeur", justify="right")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)


@app.command()
def benchmark(
    datasets: int = typer.Option(3, "--datasets"),
    out: Path = typer.Option("reports/benchmark.md", "--out"),
    seed: int = typer.Option(1234, "--seed"),
    config_path: Path = typer.Option(None, "--config", "-c"),
    separate: bool = typer.Option(None, "--separate/--no-separate"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    if separate is not None:
        cfg.separation.enabled = separate
    report = run_benchmark(datasets, seed, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    console.print(f"[green]✓[/green] rapport → {out}")


@app.command()
def viz(
    result_json: Path = typer.Argument(..., exists=True),
    out_png: Path = typer.Option("timeline.png", "--out", "-o"),
):
    from conversation_deconvolution.conversation.viz import plot_timeline
    from conversation_deconvolution.core.types import result_from_dict

    result = result_from_dict(_load(result_json))
    plot_timeline(result, out_png, title=result_json.stem)
    console.print(f"[green]✓[/green] timeline → {out_png}")


def _load(path: Path):
    import json

    return json.loads(path.read_text())


def evaluate_results(ref, hyp) -> dict:
    ref_turns = turns_of(ref)
    hyp_turns = turns_of(hyp)
    der_res = diarization_error_rate(ref_turns, hyp_turns)

    gt_utts = [u for c in ref.conversations for u in c.utterances]
    pred_utts = [u for c in hyp.conversations for u in c.utterances]
    non_overlap_ids = {
        u.id for u in gt_utts if sum(_overlap(u, v) > 0.2 for v in gt_utts if v is not u) == 0
    }
    matched = match_by_iou(gt_utts, pred_utts, min_iou=0.2)
    matched_non_overlap = [(g, p) for g, p in matched if g.id in non_overlap_ids]
    wer_metrics = (
        wer_report([(g.text, p.text) for g, p in matched_non_overlap])
        if matched_non_overlap
        else {"wer": 1.0}
    )
    matched_ov = [(g, p) for g, p in matched if g.id not in non_overlap_ids]
    wer_overlap = (
        wer_report([(g.text, p.text) for g, p in matched_ov])["wer"] if matched_ov else None
    )

    keys = {g.id: p.id for g, p in matched}
    conv_metrics = (
        conversation_metrics(ref.conversations, hyp.conversations, keys)
        if keys
        else {"pairwise_f1": 0.0, "ari": 0.0, "nmi": 0.0}
    )
    out = {
        "DER": der_res.der,
        "WER (non-overlap)": wer_metrics["wer"],
        "pairwise_F1": conv_metrics["pairwise_f1"],
        "ARI": conv_metrics["ari"],
        "NMI": conv_metrics["nmi"],
        "matched utterances": float(len(matched)),
    }
    if wer_overlap is not None:
        out["WER (overlap)"] = wer_overlap
    return out


def turns_of(result):
    from conversation_deconvolution.core.types import SpeakerTurn

    return sorted(
        [
            SpeakerTurn(u.speaker or "?", u.start, u.end)
            for c in result.conversations
            for u in c.utterances
        ],
        key=lambda t: t.start,
    )


def _overlap(a, b) -> float:
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def run_benchmark(n_datasets: int, base_seed: int, cfg: PipelineConfig) -> str:
    from conversation_deconvolution.audio.loader import load_audio
    from conversation_deconvolution.core.types import result_from_dict
    from conversation_deconvolution.pipeline import build_pipeline
    from conversation_deconvolution.synthetic.generator import SyntheticGenerator
    from conversation_deconvolution.synthetic.tts import PiperTts

    generator = SyntheticGenerator(PiperTts(), cfg.synthetic)
    n_speakers = 2 * 2
    cfg.diarization.num_speakers = n_speakers
    pipeline = build_pipeline(cfg)
    rows = []
    for k in range(n_datasets):
        ds_dir = Path(f"data/synthetic/bench_{base_seed}_{k}")
        gen_dir = generator.generate(ds_dir, seed=base_seed + k)
        result = pipeline.run(load_audio(gen_dir / "mixed.wav"))
        gt_result = result_from_dict(_load(gen_dir / "ground_truth.json"))
        metrics = evaluate_results(gt_result, result)
        rows.append(metrics)
    header = ["DER", "WER (non-overlap)", "pairwise_F1", "ARI", "NMI"]
    lines = [
        "# Benchmark — Conversation Deconvolution",
        "",
        f"- datasets : {n_datasets}",
        f"- locuteurs (oracle) : {n_speakers}",
        f"- seeds : {base_seed}…{base_seed + n_datasets - 1}",
        "",
        "| Métrique | moyenne | écart-type |",
        "|---|---|---|",
    ]
    for h in header:
        values = [r[h] for r in rows]
        lines.append(f"| {h} | {np.mean(values):.4f} | {np.std(values):.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    app()
