"""Part 1 — fine-tune pre-trained Whisper on Thai speech (FLEURS th_th).

    python -m src.train_asr                                   # whisper-tiny, fits a 4 GB GPU (8 x 2 accum)
    python -m src.train_asr --model openai/whisper-base --batch-size 4 --grad-accum 4 \
        --gradient-checkpointing
    python -m src.train_asr --model openai/whisper-small --lora --tts-repeat 4 \
        --batch-size 4 --grad-accum 4 --gradient-checkpointing --out models/whisper-small-th  # fits 4 GB
    python -m src.train_asr --resume                          # continue after an interruption
    python -m src.train_asr --max-steps 20 --eval-steps 10 --eval-limit 16 --out models/whisper-smoke

Before/after comparison:
    python -m src.asr eval --model openai/whisper-tiny --data fleurs
    python -m src.asr eval --model models/whisper-th   --data fleurs

--own-repeat N adds your own recordings from data/audio_train (src.record --set train) N times.
--tts-repeat N adds gTTS audio of the in-domain command sentences (data/raw/tts, made by
src.augment_asr; never the recorded test set) N times, with random speed/noise perturbation,
so the model sees command vocabulary and code-mixed English ("task", "calendar").

Outputs: models/whisper-th/ (best checkpoint by dev CER, + processor; LoRA merged into the weights),
         reports/asr/train_history.json
"""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Audio, Dataset, concatenate_datasets
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperForConditionalGeneration, WhisperProcessor)

from .asr import REPORT_DIR, SR, cer, decode_audio, load_fleurs
from .augment_asr import TTS_DIR, perturb
from .data import CUSTOM_CSVS, OWN_TRAIN_CSV, ROOT, read_intent_csv

MAX_AUDIO_SEC = 30          # Whisper's input window
MAX_LABEL_TOKENS = 448      # Whisper decoder max length
# Learning rates suggested for Whisper fine-tuning (smaller model -> larger LR).
DEFAULT_LR = {"tiny": 3.75e-5, "base": 2.5e-5, "small": 1.25e-5}
LORA_LR = 1e-3


@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int
    dtype: torch.dtype = torch.float32      # match the model (fp16 base when using LoRA)

    def __call__(self, features):
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(self.dtype)
        labels = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features],
                                              return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        # The model prepends <|startoftranscript|> itself; drop it if the tokenizer added it too.
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def clips(files, texts, repeat, features):
    """Audio files + transcripts, repeated, flagged for random perturbation, in FLEURS' schema."""
    return Dataset.from_dict({
        "audio": [str(f) for f in files] * repeat,
        "raw_transcription": list(texts) * repeat,
        "num_samples": [0] * len(files) * repeat,
        "augment": [True] * len(files) * repeat,
    }).cast_column("audio", Audio(decode=False)).cast(features)


def load_train(tts_repeat, own_repeat):
    """FLEURS train + gTTS command clips (×tts_repeat) + the user's own recordings (×own_repeat)."""
    fleurs = load_fleurs("train").select_columns(["audio", "raw_transcription", "num_samples"])
    fleurs = fleurs.add_column("augment", [False] * len(fleurs))
    parts = [fleurs]
    if tts_repeat:
        rows = [r for path in CUSTOM_CSVS for r in read_intent_csv(path)]
        files = [TTS_DIR / f"{i:04d}.mp3" for i in range(len(rows))]
        if not all(f.exists() for f in files):
            raise FileNotFoundError(f"TTS clips missing in {TTS_DIR} — run `python -m src.augment_asr` first")
        parts.append(clips(files, [r["text"] for r in rows], tts_repeat, fleurs.features))
    if own_repeat:
        with open(OWN_TRAIN_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        parts.append(clips([OWN_TRAIN_CSV.parent / r["file"] for r in rows], [r["text"] for r in rows],
                           own_repeat, fleurs.features))
        print(f"own recordings: {len(rows)} x {own_repeat}")
    return concatenate_datasets(parts).shuffle(seed=0)


def prepare(ds, processor):
    """Drop too-long examples, then compute log-mel features + labels lazily per batch."""
    tok = processor.tokenizer

    def keep(num_samples, texts):
        lens = [len(x) for x in tok(texts).input_ids]
        return [n <= MAX_AUDIO_SEC * SR and l <= MAX_LABEL_TOKENS for n, l in zip(num_samples, lens)]

    ds = ds.filter(keep, batched=True, input_columns=["num_samples", "raw_transcription"])

    def transform(batch):
        wavs = [decode_audio(a) for a in batch["audio"]]
        if "augment" in batch:
            rng = np.random.default_rng()
            wavs = [perturb(w, rng) if aug else w for w, aug in zip(wavs, batch["augment"])]
        feats = processor.feature_extractor(wavs, sampling_rate=SR).input_features
        return {"input_features": feats, "labels": tok(batch["raw_transcription"]).input_ids}

    return ds.with_transform(transform)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openai/whisper-tiny")
    p.add_argument("--out", default=str(ROOT / "models" / "whisper-th"))
    p.add_argument("--max-steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, help="default depends on model size")
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--eval-limit", type=int, default=200, help="dev utterances used for CER during training")
    p.add_argument("--gradient-checkpointing", action="store_true", help="less GPU memory, slower")
    p.add_argument("--no-spec-augment", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora", action="store_true",
                   help="train LoRA adapters only (lets whisper-small fit a 4 GB GPU); merged on save")
    p.add_argument("--tts-repeat", type=int, default=0,
                   help="add gTTS command clips this many times (0 = FLEURS only)")
    p.add_argument("--own-repeat", type=int, default=0,
                   help="add your own training recordings (data/audio_train) this many times")
    p.add_argument("--resume", action="store_true",
                   help="continue from the latest checkpoint in <out>-ckpt (e.g. after the machine slept)")
    args = p.parse_args()

    size = next((s for s in DEFAULT_LR if s in args.model), "small")
    lr = args.lr or (LORA_LR if args.lora else DEFAULT_LR[size])
    out = Path(args.out)

    processor = WhisperProcessor.from_pretrained(args.model, language="thai", task="transcribe")
    # LoRA: the frozen base can sit in fp16 (half the memory); peft keeps the adapters in fp32.
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16 if args.lora and torch.cuda.is_available() else None)
    model.generation_config.language = "thai"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.apply_spec_augment = not args.no_spec_augment   # mask time/freq bands: helps with ~8 h of data
    model.config.mask_time_prob = 0.05
    model.config.use_cache = not args.gradient_checkpointing
    if args.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
        model.print_trainable_parameters()

    train_ds = prepare(load_train(args.tts_repeat, args.own_repeat), processor)
    dev_ds = load_fleurs("dev")
    dev_ds = prepare(dev_ds.select(range(min(args.eval_limit, len(dev_ds)))), processor)
    print(f"train {len(train_ds)} utterances, dev {len(dev_ds)}; lr {lr}")

    def compute_metrics(pred):
        label_ids = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids)
        hyps = processor.batch_decode(pred.predictions, skip_special_tokens=True)
        refs = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"cer": cer(refs, hyps)}

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out.with_name(out.name + "-ckpt")),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # works with frozen inputs (LoRA)
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        remove_unused_columns=False,        # keep raw audio for the lazy transform
        dataloader_num_workers=args.workers,
        report_to="none",
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=Collator(processor, processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
                               model.dtype),
        compute_metrics=compute_metrics, processing_class=processor,
    )

    if not args.resume:
        print("dev CER before fine-tuning:", trainer.evaluate()["eval_cer"])
    trainer.train(resume_from_checkpoint=args.resume or None)
    final = trainer.evaluate()
    print("dev CER after fine-tuning (best checkpoint):", final["eval_cer"])

    if args.lora:
        merged = trainer.model.merge_and_unload()    # plain Whisper weights: loads like any model
        merged.save_pretrained(str(out))
    else:
        trainer.save_model(str(out))
    processor.save_pretrained(str(out))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    hist_name = "train_history.json" if out.name == "whisper-th" else f"{out.name}_history.json"
    (REPORT_DIR / hist_name).write_text(json.dumps(
        {"base_model": args.model, "lr": lr, "args": vars(args), "log_history": trainer.state.log_history},
        indent=2))
    print(f"saved model -> {out}")


if __name__ == "__main__":
    main()
