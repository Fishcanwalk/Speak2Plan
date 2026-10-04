"""ASR (speech -> text) with Whisper: data loading, transcription, CER/WER evaluation.

Evaluate a model (pre-trained or our fine-tuned one):
    python -m src.asr eval --model openai/whisper-tiny --data fleurs --limit 200
    python -m src.asr eval --model models/whisper-th --data fleurs
    python -m src.asr eval --model models/whisper-th --data data/audio/transcripts.csv

Transcribe files:
    python -m src.asr transcribe --model models/whisper-th data/audio/cmd01.wav
"""
import argparse
import csv
import io
import json
import re
import string
from pathlib import Path

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from pythainlp.tokenize import word_tokenize

from .data import DATA_DIR, ROOT

SR = 16000
FLEURS_DIR = DATA_DIR / "raw" / "fleurs_th"
FLEURS_SPLITS = {"train": "train", "dev": "validation", "test": "test"}
REPORT_DIR = ROOT / "reports" / "asr"


# ---------- data ----------

def load_fleurs(split):
    """FLEURS th_th split as a HF Dataset with undecoded audio bytes.

    Audio is decoded by us (soundfile) instead of `datasets`, which would need torchcodec/FFmpeg.
    Download first:  curl -L -o data/raw/fleurs_th/<split>.parquet \
        https://huggingface.co/api/datasets/google/fleurs/parquet/th_th/<split>/0.parquet
    """
    from datasets import Audio, load_dataset

    name = FLEURS_SPLITS[split]
    path = FLEURS_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — see load_fleurs() docstring to download it")
    ds = load_dataset("parquet", data_files={name: str(path)}, split=name)
    return ds.cast_column("audio", Audio(decode=False))


def decode_audio(audio):
    """`audio` = {"bytes": ..., "path": ...} or a file path -> float32 mono 16 kHz."""
    if isinstance(audio, dict):
        src = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    else:
        src = str(audio)
    try:
        wav, sr = sf.read(src, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except sf.LibsndfileError:  # e.g. m4a/mp3 from a phone -> librosa (audioread/ffmpeg)
        if isinstance(src, io.BytesIO):
            raise
        wav, sr = librosa.load(src, sr=None, mono=True)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav.astype(np.float32)


def load_eval_set(data, limit=None):
    """Return (audio_list, reference_texts, ids). `data` = 'fleurs' / 'fleurs-dev' or a CSV.

    CSV format (e.g. data/audio/transcripts.csv): file,text  — file relative to the CSV.
    """
    if data in ("fleurs", "fleurs-test", "fleurs-dev"):
        ds = load_fleurs("dev" if data == "fleurs-dev" else "test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        return ([decode_audio(a) for a in ds["audio"]], list(ds["raw_transcription"]),
                [str(i) for i in ds["id"]])
    csv_path = Path(data)
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:limit]
    return ([decode_audio(csv_path.parent / r["file"]) for r in rows], [r["text"] for r in rows],
            [r["file"] for r in rows])


# ---------- metrics ----------

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}“”‘’…ๆฯ]")


def normalize_text(text):
    """Lowercase, drop punctuation, collapse spaces (Thai has no word spaces anyway)."""
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def cer(refs, hyps):
    """Character error rate, ignoring spaces — the main metric for Thai."""
    r = [normalize_text(t).replace(" ", "") or "-" for t in refs]
    h = [normalize_text(t).replace(" ", "") for t in hyps]
    return jiwer.cer(r, h)


def wer(refs, hyps):
    """Word error rate after PyThaiNLP word segmentation of both sides."""
    def seg(t):
        return " ".join(w for w in word_tokenize(normalize_text(t).replace(" ", ""), engine="newmm")
                        if w.strip())
    return jiwer.wer([seg(t) or "-" for t in refs], [seg(t) for t in hyps])


# ---------- model ----------

class Transcriber:
    def __init__(self, model="models/whisper-th", device=None, language="thai"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model, dtype=dtype)
        self.model.to(self.device).eval()
        self.language = language

    @torch.no_grad()
    def transcribe(self, audios, batch_size=8, prompt=None):
        """`audios`: list of float32 16 kHz arrays or file paths. Returns list of strings.

        `prompt`: text Whisper treats as preceding context, biasing it toward those spellings
        (e.g. the wake name "เคทู"). It is not included in the output.
        """
        kw = {}
        if prompt:
            kw["prompt_ids"] = torch.tensor(self.processor.get_prompt_ids(prompt), device=self.device)
        out = []
        for i in range(0, len(audios), batch_size):
            batch = [a if isinstance(a, np.ndarray) else decode_audio(a) for a in audios[i: i + batch_size]]
            feats = self.processor.feature_extractor(batch, sampling_rate=SR, return_tensors="pt")
            feats = feats.input_features.to(self.device, dtype=self.model.dtype)
            ids = self.model.generate(feats, language=self.language, task="transcribe",
                                      max_new_tokens=225, **kw)
            out += self.processor.batch_decode(ids, skip_special_tokens=True)
        return [t.strip() for t in out]


# ---------- CLI ----------

def cmd_eval(args):
    audios, refs, ids = load_eval_set(args.data, args.limit)
    print(f"Evaluating {args.model} on {args.data} ({len(audios)} utterances)")
    hyps = Transcriber(args.model).transcribe(audios, args.batch_size)
    result = {"model": args.model, "data": args.data, "n": len(refs),
              "cer": cer(refs, hyps), "wer": wer(refs, hyps),
              "samples": [{"id": i, "ref": r, "hyp": h} for i, r, h in zip(ids, refs, hyps)]}
    print(f"CER {result['cer']:.4f}   WER {result['wer']:.4f}")
    for s in result["samples"][:5]:
        print(f"  REF: {s['ref']}\n  HYP: {s['hyp']}\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{Path(args.model).name}__{Path(args.data).stem}"
    (REPORT_DIR / f"{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved reports/asr/{tag}.json")


def cmd_transcribe(args):
    t = Transcriber(args.model)
    for f, text in zip(args.files, t.transcribe(args.files)):
        print(f"{f}\t{text}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", default="models/whisper-th")
    e.add_argument("--data", default="fleurs", help="fleurs | fleurs-dev | path/to/transcripts.csv")
    e.add_argument("--limit", type=int)
    e.add_argument("--batch-size", type=int, default=8)
    e.add_argument("--tag", help="report file name (default: <model>__<data>)")
    t = sub.add_parser("transcribe")
    t.add_argument("--model", default="models/whisper-th")
    t.add_argument("files", nargs="+")
    args = p.parse_args()
    {"eval": cmd_eval, "transcribe": cmd_transcribe}[args.cmd](args)


if __name__ == "__main__":
    main()
