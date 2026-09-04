from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import typer
from rich import box
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
    reconstructor: str = typer.Option(None, "--reconstructor", "-r"),
    diarization_backend: str = typer.Option("custom", "--diarization-backend"),
    separation_backend: str | None = typer.Option(None, "--separation-backend"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    if num_speakers:
        cfg.diarization.num_speakers = num_speakers
    if separate is not None:
        cfg.separation.enabled = separate
    if reconstructor is not None:
        cfg.reconstructor_kind = reconstructor
    if diarization_backend not in ("custom", "pyannote"):
        raise typer.BadParameter("--diarization-backend: custom|pyannote")
    cfg.diarization.backend = diarization_backend
    if separation_backend is not None:
        if separation_backend not in ("passthrough", "sepformer", "tse"):
            raise typer.BadParameter("--separation-backend: passthrough|sepformer|tse")
        cfg.separation.backend = separation_backend
        cfg.separation.enabled = separation_backend != "passthrough"
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
def demo(
    input: Path = typer.Argument(..., exists=True),
    output: Path = typer.Option("out.json", "--output", "-o"),
    config_path: Path = typer.Option(None, "--config", "-c"),
    num_speakers: int = typer.Option(None, "--num-speakers", "-n"),
    plot: Path = typer.Option(None, "--plot", "-p"),
    separate: bool = typer.Option(None, "--separate/--no-separate"),
    reconstructor: str = typer.Option(None, "--reconstructor", "-r"),
    diarization_backend: str = typer.Option("custom", "--diarization-backend"),
):
    """Démonstration interactive : traiter un fichier audio et afficher
    un résumé complet des résultats (locuteurs, timeline, transcriptions,
    conversations, export)."""
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    if num_speakers:
        cfg.diarization.num_speakers = num_speakers
    if separate is not None:
        cfg.separation.enabled = separate
    if reconstructor is not None:
        cfg.reconstructor_kind = reconstructor
    if diarization_backend not in ("custom", "pyannote"):
        raise typer.BadParameter("--diarization-backend: custom|pyannote")
    cfg.diarization.backend = diarization_backend
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

    # Tableau récapitulatif des locuteurs
    speaker_table = Table(title="Locuteurs", show_header=True, box=box.SQUARE)
    speakers = sorted(
        {u.speaker for c in result.conversations for u in c.utterances if u.speaker}
    )
    speaker_table.add_column("Locuteur")
    speaker_table.add_column("Nombre d'interventions")
    for spk in speakers:
        count = sum(1 for c in result.conversations for u in c.utterances if u.speaker == spk)
        speaker_table.add_row(spk, str(count))
    console.print(speaker_table)

    # Tableau des conversations
    conv_table = Table(title="Conversations", show_header=True, box=box.SQUARE)
    conv_table.add_column("ID")
    conv_table.add_column("Participants")
    conv_table.add_column("Utterances")
    for i, conv in enumerate(result.conversations, start=1):
        participants = ", ".join(conv.participants) if conv.participants else "Inconnu"
        n_utt = len(conv.utterances)
        conv_table.add_row(f"conversation_{i:02d}", participants, str(n_utt))
    console.print(conv_table)

    # Résumé des transcriptions
    trans_table = Table(title="Transcriptions sélectionnées", show_header=True, box=box.SQUARE)
    trans_table.add_column("Utterance")
    trans_table.add_column("Locuteur")
    trans_table.add_column("Timestamps")
    trans_table.add_column("Texte")
    for conv in result.conversations:
        for u in conv.utterances[:3]:  # first 3 per conversation
            tstamp = f"{u.start:.1f}s–{u.end:.1f}s"
            trans_table.add_row(u.text[:60], u.speaker or "?", tstamp, "")
    console.print(trans_table)

    console.print(f"\n[italic]Résultats exportés vers[/italic] [bold]{output}[/bold]")


@app.command()
def synth(
    out_dir: Path = typer.Option("data/synthetic/sample", "--out", "-o"),
    conversations: int = typer.Option(2, "--conversations"),
    speakers: int = typer.Option(2, "--speakers"),
    seed: int = typer.Option(0, "--seed"),
    snr_db: float = typer.Option(None, "--snr-db"),
    config_path: Path = typer.Option(None, "--config", "-c"),
    tts_backend: str = typer.Option("piper", "--tts-backend"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    if snr_db is not None:
        cfg.synthetic.snr_db = snr_db
    cfg.synthetic.tts_backend = tts_backend
    from conversation_deconvolution.synthetic.generator import SyntheticGenerator
    from conversation_deconvolution.synthetic.tts import create_tts

    gen = SyntheticGenerator(create_tts(cfg.synthetic.tts_backend), cfg.synthetic)
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
    reconstructor: str = typer.Option("heuristic", "--reconstructor"),
    diarization_backend: str = typer.Option("custom", "--diarization-backend"),
    separation_backend: str | None = typer.Option(None, "--separation-backend"),
    tts_backend: str = typer.Option("piper", "--tts-backend"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    if separate is not None:
        cfg.separation.enabled = separate
    if diarization_backend not in ("custom", "pyannote"):
        raise typer.BadParameter("--diarization-backend: custom|pyannote")
    cfg.diarization.backend = diarization_backend
    if separation_backend is not None:
        if separation_backend not in ("passthrough", "sepformer", "tse"):
            raise typer.BadParameter("--separation-backend: passthrough|sepformer|tse")
        cfg.separation.backend = separation_backend
        cfg.separation.enabled = separation_backend != "passthrough"
    cfg.synthetic.tts_backend = tts_backend
    kinds = {
        "both": ["heuristic", "graph"],
        "heuristic": ["heuristic"],
        "graph": ["graph"],
    }.get(reconstructor)
    if kinds is None:
        raise typer.BadParameter("--reconstructor: heuristic|graph|both")
    report = run_benchmark(datasets, seed, cfg, kinds)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    console.print(f"[green]✓[/green] rapport → {out}")


@app.command()
def train(
    datasets: int = typer.Option(8, "--datasets"),
    out: Path = typer.Option("models/graph_lr.json", "--out"),
    seed_base: int = typer.Option(3000, "--seed-base"),
    conversations: int = typer.Option(2, "--conversations"),
    speakers: int = typer.Option(2, "--speakers"),
    config_path: Path = typer.Option(None, "--config", "-c"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    from conversation_deconvolution.conversation.pair_features import (
        pair_feature_names,
    )
    from conversation_deconvolution.conversation.semantic import (
        SentenceTransformerEmbedder,
    )
    from conversation_deconvolution.conversation.trainer import (
        build_training_set,
        fit_edge_classifier,
        save_model,
    )
    from conversation_deconvolution.synthetic.generator import SyntheticGenerator
    from conversation_deconvolution.synthetic.tts import create_tts

    generator = SyntheticGenerator(create_tts(cfg.synthetic.tts_backend), cfg.synthetic)
    dirs = []
    for k in range(datasets):
        target = Path(f"data/synthetic/train_{seed_base}_{k}")
        generator.generate(
            target,
            seed=seed_base + k,
            n_conversations=conversations,
            speakers_per_thread=speakers,
        )
        dirs.append(target)
        console.print(f"[green]✓[/green] dataset {k + 1}/{datasets} → {target}")
    embedder = SentenceTransformerEmbedder(cfg.text_embedding_model)
    X, y = build_training_set(dirs, embedder, cfg.graph, rng_seed=cfg.graph.seed)
    model = fit_edge_classifier(X, y, pair_feature_names(), seed=cfg.graph.seed)
    model["meta"].update(
        {
            "n_datasets": datasets,
            "seed_base": seed_base,
            "negative_ratio": cfg.graph.negative_ratio,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    save_model(model, out)
    console.print(
        f"[green]✓[/green] classifieur entraîné "
        f"({len(y)} paires, F1 arêtes CV={model['meta']['pairwise_cv_f1']:.3f}) → {out}"
    )


@app.command()
def train_tse(
    datasets: int = typer.Option(8, "--datasets"),
    epochs: int = typer.Option(30, "--epochs"),
    out: Path = typer.Option(Path("models/tse/model.pt"), "--out"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    seed_base: int = typer.Option(3000, "--seed-base"),
):
    cfg = PipelineConfig.from_yaml(config_path) if config_path else PipelineConfig.default()
    cfg.tse.epochs = epochs
    from conversation_deconvolution.synthetic.tts import create_tts
    from conversation_deconvolution.tse.dataset import TseDataset
    from conversation_deconvolution.tse.train import train_tse_model

    tts = create_tts(cfg.synthetic.tts_backend)
    dirs = sorted(Path("data/synthetic").glob(f"train_{seed_base}_*"))
    if not dirs:
        dirs = sorted(Path("data/synthetic").glob("train_3000_*"))
    if not dirs:
        console.print(
            "[red]✗[/red] aucun dataset train_3000_* trouvé — générez-en avec"
            " `deconvolute synth`"
        )
        return
    if len(dirs) > datasets:
        dirs = dirs[:datasets]
    dataset = TseDataset(tts, cfg.tse, dirs)
    model_path = train_tse_model(dataset, cfg.tse, str(out))
    console.print(f"[green]✓[/green] modèle TSE → {model_path}")


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
        else None
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
        "WER (non-overlap)": wer_metrics["wer"] if wer_metrics else None,
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


def run_benchmark(
    n_datasets: int, base_seed: int, cfg: PipelineConfig, kinds: list[str]
) -> str:
    from conversation_deconvolution.audio.loader import load_audio
    from conversation_deconvolution.core.types import result_from_dict
    from conversation_deconvolution.pipeline import build_pipeline
    from conversation_deconvolution.synthetic.generator import SyntheticGenerator
    from conversation_deconvolution.synthetic.tts import create_tts

    generator = SyntheticGenerator(create_tts(cfg.synthetic.tts_backend), cfg.synthetic)
    n_speakers = 2 * 2
    cfg.diarization.num_speakers = n_speakers
    ds_dirs = []
    for k in range(n_datasets):
        ds_dir = Path(f"data/synthetic/bench_{base_seed}_{k}")
        ds_dirs.append(generator.generate(ds_dir, seed=base_seed + k))
    lines = [
        "# Benchmark — Conversation Deconvolution",
        "",
        f"- datasets : {n_datasets}",
        f"- locuteurs (oracle) : {n_speakers}",
        f"- seeds : {base_seed}…{base_seed + n_datasets - 1}",
        "",
    ]
    for kind in kinds:
        cfg.reconstructor_kind = kind
        pipeline = build_pipeline(cfg)
        rows = []
        for gen_dir in ds_dirs:
            result = pipeline.run(load_audio(gen_dir / "mixed.wav"))
            gt_result = result_from_dict(_load(gen_dir / "ground_truth.json"))
            rows.append(evaluate_results(gt_result, result))
        lines += format_section(kind, rows)
        lines.append("")
    return "\n".join(lines) + "\n"


def format_section(kind: str, rows: list[dict]) -> list[str]:
    header = ["DER", "WER (non-overlap)", "pairwise_F1", "ARI", "NMI"]
    if any(r.get("WER (overlap)") is not None for r in rows):
        header.insert(2, "WER (overlap)")
    out = [
        f"## Reconstruteur : {kind}",
        "",
        "| Métrique | moyenne | écart-type |",
        "|---|---|---|",
    ]
    for h in header:
        values = [r[h] for r in rows if r.get(h) is not None]
        if not values:
            out.append(f"| {h} | - | - |")
        else:
            out.append(f"| {h} | {np.mean(values):.4f} | {np.std(values):.4f} |")
    return out


if __name__ == "__main__":
    app()
