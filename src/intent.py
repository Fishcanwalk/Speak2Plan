"""Load a trained intent classifier and predict.

    python -m src.intent --model cnn "เช็ค task ใน google ให้หน่อย"
"""
import argparse

import joblib
import torch

from .data import ROOT, tokenize
from .models import MODELS, Vocab, pad_batch

MODEL_DIR = ROOT / "models"


class IntentClassifier:
    def __init__(self, name="cnn", device=None):
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sklearn = (MODEL_DIR / f"intent_{name}.joblib").exists()
        if self.sklearn:
            bundle = joblib.load(MODEL_DIR / f"intent_{name}.joblib")
            self.vec, self.clf, self.labels = bundle["vectorizer"], bundle["clf"], bundle["labels"]
            self.joined = bundle.get("joined", False)
        else:
            ckpt = torch.load(MODEL_DIR / f"intent_{name}.pt", map_location="cpu")
            self.labels, self.max_len = ckpt["labels"], ckpt["max_len"]
            self.vocab = Vocab(ckpt["vocab"])
            self.model = MODELS[ckpt["model"]](len(self.vocab), len(self.labels), **ckpt["config"])
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.to(self.device).eval()
            self.min_len = max(getattr(self.model, "kernel_sizes", (1,)))

    @torch.no_grad()
    def predict_proba(self, texts):
        """Return list of {label: prob} dicts."""
        tokens = [tokenize(t) for t in texts]
        if self.sklearn:
            X = [" ".join(t) for t in tokens] if self.joined else tokens
            probs = self.clf.predict_proba(self.vec.transform(X))
        else:
            ids, lengths = pad_batch([self.vocab.encode(t, self.max_len) for t in tokens], self.min_len)
            probs = torch.softmax(self.model(ids.to(self.device), lengths), dim=1).cpu().numpy()
        return [dict(zip(self.labels, map(float, p))) for p in probs]

    def predict(self, texts):
        return [max(p, key=p.get) for p in self.predict_proba(texts)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn", choices=["nb", "logreg", "logreg_char", "cnn", "lstm"])
    p.add_argument("texts", nargs="+")
    args = p.parse_args()
    clf = IntentClassifier(args.model)
    for text, probs in zip(args.texts, clf.predict_proba(args.texts)):
        best = max(probs, key=probs.get)
        print(f"{best:15s} {probs[best]:.3f}  {text}")


if __name__ == "__main__":
    main()
