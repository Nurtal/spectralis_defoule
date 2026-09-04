"""AMI corpus extraction and fabrication for parallel conversation benchmarks.

Downloads AMI meeting segments from HuggingFace, groups by speaker,
and fabricates parallel-conversation audio by superimposing tracks
from different meetings.

Usage:
    uv run python -m conversation_deconvolution.synthetic.ami_fabricate \
        --output data/ami_fabricated \
        --n-fabrications 4 \
        --seed 42
"""
import argparse
import io
import json
import struct
import wave
from pathlib import Path

import numpy as np


def load_ami_segments(max_shards: int = 2) -> list[dict]:
    """Load AMI segments from HuggingFace ihm split."""
    from huggingface_hub import list_repo_files

    files = list_repo_files("edinburghcstr/ami", repo_type="dataset")
    shard_files = [
        f for f in files if f.startswith("ihm/") and f.endswith(".parquet")
    ][:max_shards]

    segments = []
    for shard in shard_files:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("edinburghcstr/ami", shard, repo_type="dataset")
        table = pq.read_table(path)
        df = table.to_pandas()
        for _, row in df.iterrows():
            segments.append({
                "meeting_id": row["meeting_id"],
                "speaker_id": row["speaker_id"],
                "audio_bytes": row["audio"]["bytes"],
                "text": row["text"],
                "begin_time": row["begin_time"],
                "end_time": row["end_time"],
            })
    return segments


def wav_bytes_to_np(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert WAV bytes to numpy float32 array + sample rate."""
    pos = 0
    riff, size, wave_id = struct.unpack_from("<4sI4s", wav_bytes, pos)
    pos += 12
    sr = 16000
    n_channels = 1
    bits_per_sample = 16
    fmt_tag = 1
    data = b""
    while pos < len(wav_bytes) - 8:
        chunk_id, chunk_size = struct.unpack_from("<4sI", wav_bytes, pos)
        pos += 8
        if chunk_id == b"fmt ":
            fmt_tag, n_channels, sr, byte_rate, block_align, bits_per_sample = (
                struct.unpack_from("<HHIIHH", wav_bytes, pos)
            )
            pos += chunk_size
        elif chunk_id == b"data":
            data = wav_bytes[pos : pos + chunk_size]
            break
        else:
            pos += chunk_size

    if not data:
        raise ValueError("No data chunk found in WAV")

    if fmt_tag == 3:
        arr = np.frombuffer(data, dtype=np.float32)
    elif bits_per_sample == 16:
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif bits_per_sample == 32:
        arr = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels)[:, 0]
    return arr, sr


def group_by_speaker(segments: list[dict]) -> dict[str, list[dict]]:
    """Group segments by speaker_id, sorted by begin_time."""
    speakers: dict[str, list[dict]] = {}
    for seg in segments:
        speakers.setdefault(seg["speaker_id"], []).append(seg)
    for spk in speakers:
        speakers[spk].sort(key=lambda s: s["begin_time"])
    return speakers


def fabricate_parallel(
    speakers: dict[str, list[dict]],
    n_conversations: int = 2,
    speakers_per_conversation: int = 2,
    rng: np.random.Generator = None,
    max_segments_per_speaker: int = 15,
    sr: int = 16000,
) -> tuple[np.ndarray, dict]:
    """Superimpose speaker segments to create parallel conversations.
    Returns (mixed_audio, ground_truth_dict)."""
    if rng is None:
        rng = np.random.default_rng()

    all_spk = list(speakers.keys())
    rng.shuffle(all_spk)
    total_spk_needed = n_conversations * speakers_per_conversation
    if not all_spk:
        raise ValueError("No valid speakers available")
    if len(all_spk) < total_spk_needed:
        all_spk = (all_spk * ((total_spk_needed // len(all_spk)) + 1))[:total_spk_needed]

    conversations = []
    all_utt_events = []  # (time_sec, spk_id, audio_np, text)
    global_offset = 0.0

    for conv_idx in range(n_conversations):
        spk_ids = all_spk[
            conv_idx * speakers_per_conversation : (conv_idx + 1) * speakers_per_conversation
        ]
        conv_start = rng.uniform(0.5, 2.0)
        global_offset += conv_start
        conv_utts = []

        for spk_id in spk_ids:
            segs = speakers[spk_id][:max_segments_per_speaker]
            local_offset = 0.0
            for seg in segs:
                audio, seg_sr = wav_bytes_to_np(seg["audio_bytes"])
                if seg_sr != sr:
                    from scipy.signal import resample_poly
                    gcd = int(np.gcd(seg_sr, sr))
                    audio = resample_poly(audio, sr // gcd, seg_sr // gcd)

                utt_start = global_offset + local_offset
                utt_end = utt_start + len(audio) / sr
                text = seg["text"].strip()
                if not text:
                    continue

                all_utt_events.append((utt_start, spk_id, audio, text))
                conv_utts.append({
                    "speaker": spk_id,
                    "start": round(utt_start, 4),
                    "end": round(utt_end, 4),
                    "text": text,
                })
                gap = seg["end_time"] - seg["begin_time"]
                local_offset += len(audio) / sr + max(0.15, gap)

        conversations.append({
            "id": f"conversation_{conv_idx + 1:02d}",
            "participants": spk_ids,
            "utterances": conv_utts,
        })
        max_utt_end = max(u["end"] for u in conv_utts) if conv_utts else global_offset
        global_offset = max_utt_end + rng.uniform(1.0, 3.0)

    total_duration = global_offset + 1.0
    mixed = np.zeros(int(total_duration * sr), dtype=np.float32)
    for utt_start, spk_id, audio, text in all_utt_events:
        start_sample = int(utt_start * sr)
        end_sample = min(start_sample + len(audio), len(mixed))
        actual_len = end_sample - start_sample
        mixed[start_sample:end_sample] += audio[:actual_len]

    peak = float(np.max(np.abs(mixed))) or 1.0
    if peak > 0.95:
        mixed *= 0.95 / peak

    gt = {
        "conversations": conversations,
        "sample_rate": sr,
        "source": "ami_fabricated",
    }
    return mixed, gt


def save_wav(audio: np.ndarray, path: Path, sr: int = 16000):
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="AMI fabricate parallel conversations")
    parser.add_argument("--output", type=str, default="data/ami_fabricated")
    parser.add_argument("--n-fabrications", type=int, default=4)
    parser.add_argument("--n-conversations", type=int, default=2)
    parser.add_argument("--speakers-per-conversation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-shards", type=int, default=4)
    parser.add_argument("--max-segments-per-speaker", type=int, default=15)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading AMI segments...")
    segments = load_ami_segments(max_shards=args.max_shards)
    print(f"  {len(segments)} segments loaded")

    speakers = group_by_speaker(segments)
    print(f"  {len(speakers)} speakers found")

    rng = np.random.default_rng(args.seed)
    for k in range(args.n_fabrications):
        fabricate_rng = np.random.default_rng(args.seed + k)
        mixed, gt = fabricate_parallel(
            speakers,
            n_conversations=args.n_conversations,
            speakers_per_conversation=args.speakers_per_conversation,
            rng=fabricate_rng,
            max_segments_per_speaker=args.max_segments_per_speaker,
        )
        ds_dir = out_dir / f"fabric_{args.seed}_{k}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        save_wav(mixed, ds_dir / "mixed.wav")
        (ds_dir / "ground_truth.json").write_text(json.dumps(gt, indent=2, default=str))
        n_utts = sum(len(c["utterances"]) for c in gt["conversations"])
        print(f"  [{k+1}/{args.n_fabrications}] {ds_dir} ({len(mixed)/16000:.1f}s, {n_utts} utts)")


if __name__ == "__main__":
    main()
