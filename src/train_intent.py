"""Part 2 — train the intent classifier from scratch and compare models.

    python -m src.train_intent                      # all: nb logreg cnn lstm
    python -m src.train_intent --models cnn lstm --epochs 40

Outputs:
    models/intent_{nb,logreg}.joblib, models/intent_{cnn,lstm}.pt
    reports/intent/{metrics.json, <model>_report.txt, <model>_confusion.png, <model>_history.json}
"""
import argparse
import json
import random
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion

from .data import LABELS, ROOT, load_intent_data, tokenize, unigrams_bigrams, word_analyzer
from .models import MODELS, Vocab, pad_batch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "intent"
MAX_LEN = 40
BASELINES = ["nb", "logreg", "logreg_char"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(name, y_true, y_pred, test_rows):
    """Print + save metrics, report, and confusion matrix. Returns metrics dict."""
    labels = list(range(len(LABELS)))
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    for loc in sorted({r["locale"] for r in test_rows}):
        idx = [i for i, r in enumerate(test_rows) if r["locale"] == loc]
        metrics[f"accuracy_{loc}"] = accuracy_score([y_true[i] for i in idx], [y_pred[i] for i in idx])

    report = classification_report(y_true, y_pred, labels=labels, target_names=LABELS, digits=3,
                                   zero_division=0)
    (REPORT_DIR / f"{name}_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{name} — acc {metrics['accuracy']:.3f}, macro-F1 {metrics['macro_f1']:.3f}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"{name}_confusion.png", dpi=120)
    plt.close()

    print(f"\n=== {name} ===")
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))
    print(report)
    return metrics


# ---------- Baselines: TF-IDF + Naive Bayes / Logistic Regression ----------

def make_baseline(name):
    """Returns (vectorizer, classifier, joined). `joined`: vectorizer takes space-joined tokens."""
    if name == "logreg_char":
        # word uni/bigrams + character 2–4-grams: robust to ASR misspellings ("task" vs "ทัก")
        vec = FeatureUnion([("word", TfidfVectorizer(analyzer=word_analyzer, sublinear_tf=True)),
                            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                                     sublinear_tf=True))])
        return vec, LogisticRegression(max_iter=3000, C=10), True
    vec = TfidfVectorizer(analyzer=unigrams_bigrams)
    clf = MultinomialNB(alpha=0.1) if name == "nb" else LogisticRegression(max_iter=2000, C=10)
    return vec, clf, False


def train_baseline(name, splits, tokens, y):
    vec, clf, joined = make_baseline(name)
    X = {s: [" ".join(t) for t in tokens[s]] if joined else tokens[s] for s in ("train", "test")}
    X_train = vec.fit_transform(X["train"])
    X_test = vec.transform(X["test"])
    t0 = time.time()
    clf.fit(X_train, y["train"])
    print(f"[{name}] trained in {time.time() - t0:.1f}s")
    joblib.dump({"vectorizer": vec, "clf": clf, "labels": LABELS, "joined": joined},
                MODEL_DIR / f"intent_{name}.joblib")
    return evaluate(name, y["test"], clf.predict(X_test).tolist(), splits["test"])


# ---------- Neural: TextCNN / BiLSTM ----------

def batches(ids, y, batch_size, shuffle, min_len):
    order = list(range(len(ids)))
    if shuffle:
        random.shuffle(order)
    for i in range(0, len(order), batch_size):
        idx = order[i: i + batch_size]
        x, lengths = pad_batch([ids[j] for j in idx], min_len)
        yield x, lengths, torch.tensor([y[j] for j in idx])


@torch.no_grad()
def predict(model, ids, device, min_len, batch_size=256):
    model.eval()
    preds = []
    for x, lengths, _ in batches(ids, [0] * len(ids), batch_size, False, min_len):
        preds += model(x.to(device), lengths).argmax(1).tolist()
    return preds


def train_neural(name, splits, tokens, y, args, device):
    vocab = Vocab.build(tokens["train"], min_freq=args.min_freq)
    ids = {s: [vocab.encode(t, MAX_LEN) for t in tokens[s]] for s in tokens}
    config = {"emb_dim": args.emb_dim, "dropout": args.dropout}
    model = MODELS[name](len(vocab), len(LABELS), **config).to(device)
    min_len = max(getattr(model, "kernel_sizes", (1,)))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"[{name}] vocab={len(vocab)} params={sum(p.numel() for p in model.parameters()):,}")

    best_f1, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, lengths, yb in batches(ids["train"], y["train"], args.batch_size, True, min_len):
            opt.zero_grad()
            loss = loss_fn(model(x.to(device), lengths), yb.to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        train_loss = total / len(ids["train"])
        dev_pred = predict(model, ids["dev"], device, min_len)
        dev_f1 = f1_score(y["dev"], dev_pred, average="macro")
        dev_acc = accuracy_score(y["dev"], dev_pred)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_acc": dev_acc, "dev_f1": dev_f1})
        print(f"[{name}] epoch {epoch:2d} loss {train_loss:.4f} dev acc {dev_acc:.4f} macro-F1 {dev_f1:.4f}")
        if dev_f1 > best_f1:
            best_f1, bad_epochs = dev_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[{name}] early stopping (best dev macro-F1 {best_f1:.4f})")
                break

    model.load_state_dict(best_state)
    torch.save({"model": name, "config": config, "state_dict": best_state, "vocab": vocab.itos,
                "labels": LABELS, "max_len": MAX_LEN}, MODEL_DIR / f"intent_{name}.pt")
    (REPORT_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=2))
    return evaluate(name, y["test"], predict(model, ids["test"], device, min_len), splits["test"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=BASELINES + ["cnn", "lstm"],
                   choices=BASELINES + ["cnn", "lstm"])
    p.add_argument("--other-ratio", type=float, default=0.12,
                   help="fraction of MASSIVE 'other' utterances to keep (it is ~90%% of the data)")
    p.add_argument("--custom-repeat", type=int, default=3,
                   help="oversample in-domain sentences (custom/generated/augmented) this many times")
    p.add_argument("--no-augmented", action="store_true", help="ignore data/augmented_intents.csv")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--min-freq", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = load_intent_data(other_ratio=args.other_ratio, seed=args.seed,
                              custom_repeat=args.custom_repeat, augmented=not args.no_augmented)
    for s, rows in splits.items():
        counts = {lab: sum(r["intent"] == lab for r in rows) for lab in LABELS}
        print(f"{s:5s} n={len(rows):5d} {counts}")
    tokens = {s: [tokenize(r["text"]) for r in rows] for s, rows in splits.items()}
    y = {s: [LABELS.index(r["intent"]) for r in rows] for s, rows in splits.items()}

    results = {}
    for name in args.models:
        set_seed(args.seed)
        if name in BASELINES:
            results[name] = train_baseline(name, splits, tokens, y)
        else:
            results[name] = train_neural(name, splits, tokens, y, args, device)

    path = REPORT_DIR / "metrics.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**old, **results}, indent=2))
    print("\nSummary (test):")
    for name, m in results.items():
        print(f"  {name:7s} acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
