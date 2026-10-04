"""Make intent training text that contains realistic ASR errors.

    python -m src.augment_asr
    python -m src.augment_asr --asr-models models/whisper-th openai/whisper-small

Each in-domain training sentence (custom + generated, never the recorded test set) is spoken with
gTTS, perturbed (speed, background noise), and transcribed by Whisper. The (noisy transcript,
original intent) pairs go to data/augmented_intents.csv, which src.data adds to the training split,
so the classifier learns what e.g. "เช็ค task" looks like after Whisper turns it into "เช็คทัก".
"""
import argparse
import csv
import gc
import io

import librosa
import numpy as np
import soundfile as sf
import torch
from gtts import gTTS

from .asr import SR, Transcriber
from .data import AUGMENTED_CSV, RAW_DIR, read_intent_csv, CUSTOM_CSVS

TTS_DIR = RAW_DIR / "tts"


def synth(text, i):
    """gTTS -> float32 16 kHz (cached as mp3)."""
    TTS_DIR.mkdir(parents=True, exist_ok=True)
    path = TTS_DIR / f"{i:04d}.mp3"
    if not path.exists():
        gTTS(text, lang="th").save(str(path))
    wav, sr = sf.read(str(path), dtype="float32")
    return librosa.resample(wav, orig_sr=sr, target_sr=SR)


def perturb(wav, rng):
    """Random speed (0.85–1.15x) + background noise at 10–30 dB SNR."""
    wav = librosa.effects.time_stretch(wav, rate=rng.uniform(0.85, 1.15))
    snr = rng.uniform(10, 30)
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    noise *= np.sqrt((wav ** 2).mean() / (10 ** (snr / 10))) / (noise.std() + 1e-8)
    return wav + noise


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asr-models", nargs="+", default=["models/whisper-th", "openai/whisper-small"])
    p.add_argument("--variants", type=int, default=2, help="perturbed copies per sentence")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = [r for path in CUSTOM_CSVS for r in read_intent_csv(path)]
    rng = np.random.default_rng(args.seed)
    print(f"Synthesizing {len(rows)} sentences with gTTS ...")
    clean = [synth(r["text"], i) for i, r in enumerate(rows)]
    audios, labels, origins = [], [], []
    for r, wav in zip(rows, clean):
        for v in range(args.variants):
            audios.append(wav if v == 0 else perturb(wav, rng))
            labels.append(r["intent"])
            origins.append(r["text"])

    out = []
    for m in args.asr_models:
        print(f"Transcribing {len(audios)} clips with {m} ...")
        t = Transcriber(m)
        for text, intent, orig in zip(t.transcribe(audios, batch_size=16), labels, origins):
            if text.strip():
                out.append({"text": text, "intent": intent, "source": m, "original": orig})
        del t
        gc.collect()
        torch.cuda.empty_cache()

    with open(AUGMENTED_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "intent", "source", "original"])
        w.writeheader()
        w.writerows(out)
    print(f"saved {len(out)} rows -> {AUGMENTED_CSV}")
    for r in out[:: max(1, len(out) // 8)]:
        print(f"  {r['original']}  ->  {r['text']}")


if __name__ == "__main__":
    main()
