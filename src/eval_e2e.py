"""End-to-end test on your own recordings: speech -> Whisper -> text -> intent.

    python -m src.eval_e2e
    python -m src.eval_e2e --asr-models openai/whisper-tiny models/whisper-th openai/whisper-small

For each ASR model: CER/WER vs. the reference text, then intent accuracy of every intent model
on (a) the reference text and (b) the ASR output — the gap is the cost of ASR errors.
Outputs reports/e2e/{summary.json, utterances.csv}.
"""
import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from .asr import Transcriber, cer, load_eval_set, normalize_text, wer
from .data import DATA_DIR, ROOT
from .intent import IntentClassifier

REPORT_DIR = ROOT / "reports" / "e2e"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(DATA_DIR / "audio" / "transcripts.csv"))
    p.add_argument("--asr-models", nargs="+", default=["openai/whisper-tiny", "models/whisper-th"])
    p.add_argument("--intent-models", nargs="+", default=["nb", "logreg", "logreg_char", "cnn", "lstm"])
    args = p.parse_args()

    audios, refs, files = load_eval_set(args.data)
    with open(args.data, encoding="utf-8") as f:
        gold = [r["intent"] for r in csv.DictReader(f)]
    print(f"{len(audios)} recordings from {args.data}")

    hyps = {}
    for m in args.asr_models:
        t = Transcriber(m)
        hyps[m] = t.transcribe(audios)
        del t
        gc.collect()
        torch.cuda.empty_cache()

    texts = {"reference": refs, **hyps}
    preds = {}
    for im in args.intent_models:
        clf = IntentClassifier(im)
        preds[im] = {src: clf.predict(ts) for src, ts in texts.items()}

    def acc(pred):
        return sum(p == g for p, g in zip(pred, gold)) / len(gold)

    summary = {"n": len(gold), "asr": {}, "intent_accuracy": {}}
    for m in args.asr_models:
        summary["asr"][m] = {"cer": cer(refs, hyps[m]), "wer": wer(refs, hyps[m]),
                             "exact_match": sum(normalize_text(r).replace(" ", "") == normalize_text(h).replace(" ", "")
                                                for r, h in zip(refs, hyps[m])) / len(refs)}
    for im in args.intent_models:
        summary["intent_accuracy"][im] = {src: acc(pr) for src, pr in preds[im].items()}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(REPORT_DIR / "utterances.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "gold_intent", "source", "text"] + [f"pred_{im}" for im in args.intent_models])
        for i, file in enumerate(files):
            for src, ts in texts.items():
                w.writerow([file, gold[i], Path(src).name, ts[i]] + [preds[im][src][i] for im in args.intent_models])

    print("\nASR on your recordings:")
    for m, s in summary["asr"].items():
        print(f"  {m:25s} CER {s['cer']:.3f}  WER {s['wer']:.3f}  exact {s['exact_match']:.2f}")
    print("\nIntent accuracy (text source ->):")
    print("  " + " " * 8 + "".join(f"{Path(s).name:>22s}" for s in texts))
    for im, row in summary["intent_accuracy"].items():
        print(f"  {im:8s}" + "".join(f"{row[s]:22.3f}" for s in texts))
    print(f"\nsaved {REPORT_DIR.relative_to(ROOT)}/summary.json, utterances.csv")


if __name__ == "__main__":
    main()
