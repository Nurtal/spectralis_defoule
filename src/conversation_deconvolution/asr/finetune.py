from pathlib import Path

import numpy as np
import soundfile as sf

from conversation_deconvolution.core.types import result_from_dict


def build_asr_dataset(dataset_dirs, tts=None, use_mixed=False):
    import json

    samples = []
    for d in dataset_dirs:
        gt_path = Path(d) / "ground_truth.json"
        wav_path = Path(d) / "mixed.wav"
        if not gt_path.exists() or not wav_path.exists():
            continue
        data = json.loads(gt_path.read_text())
        if use_mixed:
            import soundfile as sf2

            mixed, sr = sf2.read(str(wav_path))
            if mixed.ndim > 1:
                mixed = mixed.mean(axis=1)
        for conv in data["conversations"]:
            for utt in conv["utterances"]:
                text = utt["text"].strip()
                if not text:
                    continue
                if use_mixed:
                    s = int(float(utt["start"]) * 16000)
                    e = int(float(utt["end"]) * 16000)
                    seg = mixed[s:e]
                    if len(seg) < 1600:
                        continue
                    samples.append((seg.astype(np.float32), text))
                else:
                    if tts is None:
                        continue
                    audio, _ = tts.synthesize(text, utt["speaker"])
                    # map speaker to voice via scenario VOICES
                    from conversation_deconvolution.synthetic.scenario import VOICES

                    # use tts directly with speaker as voice? we need mapping
                    # Already tts.synthesize handles voice; we already did but need clean
                    # For clean, we use tts cache; for mixed we sliced
                    # Here we already have audio from tts.synthesize with speaker id, but that is not valid voice
                    # So we need to map speaker->voice
                    pass
    return samples


def build_clean_dataset(dataset_dirs):
    from conversation_deconvolution.synthetic.tts import PiperTts
    from conversation_deconvolution.synthetic.scenario import VOICES
    import json

    tts = PiperTts()
    # Build speaker->voice map from all datasets (sorted)
    all_speakers = set()
    for d in dataset_dirs:
        gt = json.loads((Path(d) / "ground_truth.json").read_text())
        for conv in gt["conversations"]:
            for utt in conv["utterances"]:
                all_speakers.add(utt["speaker"])
    sorted_spk = sorted(all_speakers)
    spk2voice = {spk: VOICES[i % len(VOICES)] for i, spk in enumerate(sorted_spk)}
    samples = []
    for d in dataset_dirs:
        gt = json.loads((Path(d) / "ground_truth.json").read_text())
        for conv in gt["conversations"]:
            for utt in conv["utterances"]:
                text = utt["text"].strip()
                voice = spk2voice[utt["speaker"]]
                audio, _ = tts.synthesize(text, voice)
                audio = np.asarray(audio, dtype=np.float32)
                if len(audio) < 1600:
                    continue
                samples.append((audio, text))
    return samples


def finetune_whisper(
    model_name="openai/whisper-small",
    train_dirs=None,
    val_dirs=None,
    output_dir="models/whisper-ft",
    epochs=3,
    lr=1e-5,
    batch_size=8,
):
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from torch.utils.data import Dataset, DataLoader
    import jiwer

    if train_dirs is None:
        train_dirs = sorted(Path("data/synthetic").glob("train_3000_*"))[:8]
    if val_dirs is None:
        val_dirs = sorted(Path("data/synthetic").glob("val_4000_*"))[:2]

    print(f"Building dataset from {len(train_dirs)} train dirs, {len(val_dirs)} val dirs")
    train_samples = build_clean_dataset(train_dirs)
    val_samples = build_clean_dataset(val_dirs)
    print(f"train {len(train_samples)} samples, val {len(val_samples)}")

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    # Tokenizer: set language
    processor.tokenizer.set_prefix_tokens(language="fr", task="transcribe")

    class WhisperDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            audio, text = self.samples[idx]
            # Whisper expects 30s padded to 3000 mel frames
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.squeeze(0)
            labels = processor.tokenizer(text).input_ids
            return {"input_features": input_features, "labels": torch.tensor(labels, dtype=torch.long)}

    def collate(batch):
        input_features = torch.stack([b["input_features"] for b in batch])
        labels = [b["labels"] for b in batch]
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        # replace -100 with pad token for loss?
        return {"input_features": input_features, "labels": labels_padded}

    train_ds = WhisperDataset(train_samples)
    val_ds = WhisperDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            # shift labels for decoder input?
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch+1}/{epochs} step {step}/{len(train_loader)} loss {loss.item():.4f}")
        avg = total_loss / len(train_loader)
        print(f"epoch {epoch+1} train loss {avg:.4f}")

        # val WER
        model.eval()
        hyps = []
        refs = []
        with torch.no_grad():
            for batch in val_loader:
                input_features = batch["input_features"].to(device)
                labels = batch["labels"]
                generated = model.generate(input_features, language="fr", task="transcribe")
                trans = processor.batch_decode(generated, skip_special_tokens=True)
                # decode refs
                for i, t in enumerate(trans):
                    # labels may be -100 padded, need to filter
                    ref_ids = labels[i].tolist()
                    ref_ids = [x for x in ref_ids if x != -100]
                    ref_text = processor.tokenizer.decode(ref_ids, skip_special_tokens=True)
                    hyps.append(t)
                    refs.append(ref_text)
        wer = jiwer.wer(refs, hyps) if refs else 0
        print(f"epoch {epoch+1} val WER {wer:.4f} ({len(refs)} samples)")
        model.train()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"saved to {output_dir}")
    return output_dir
