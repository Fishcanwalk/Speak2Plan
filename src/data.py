"""Intent data: load MASSIVE (th-TH + en-US), map to our 6 intents, tokenize.

MASSIVE source: https://github.com/alexa/massive (CC-BY-4.0).
The HF `AmazonScience/massive` repo uses a loading script that new `datasets`
versions no longer run, so we read the original jsonl from the release tar.
"""
import csv
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path

from pythainlp.tokenize import word_tokenize

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_TAR = RAW_DIR / "massive-1.1.tar.gz"
CUSTOM_CSV = DATA_DIR / "custom_intents.csv"

LABELS = ["check_tasks", "add_task", "complete_task", "check_calendar", "add_event", "other"]

# MASSIVE intent -> our intent. Everything not listed becomes "other".
# MASSIVE has no "to-do list" scenario; its shopping/general "lists" are the closest match.
INTENT_MAP = {
    "lists_query": "check_tasks",
    "lists_createoradd": "add_task",
    "lists_remove": "complete_task",
    "calendar_query": "check_calendar",
    "calendar_set": "add_event",
}

SPLITS = {"train": "train", "dev": "dev", "test": "test"}


def download_massive():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not MASSIVE_TAR.exists():
        print(f"Downloading MASSIVE -> {MASSIVE_TAR}")
        urllib.request.urlretrieve(MASSIVE_URL, MASSIVE_TAR)


def load_massive(locales=("th-TH", "en-US")):
    """Return list of dicts: text, intent (ours), source_intent, locale, split."""
    download_massive()
    rows = []
    with tarfile.open(MASSIVE_TAR) as tar:
        for loc in locales:
            f = tar.extractfile(f"1.1/data/{loc}.jsonl")
            for line in f:
                r = json.loads(line)
                rows.append({
                    "text": r["utt"],
                    "intent": INTENT_MAP.get(r["intent"], "other"),
                    "source_intent": r["intent"],
                    "locale": loc,
                    "split": SPLITS[r["partition"]],
                })
    return rows


def load_custom():
    """Self-written sentences (data/custom_intents.csv: text,intent). Train only."""
    if not CUSTOM_CSV.exists():
        return []
    rows = []
    with open(CUSTOM_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            text, intent = r["text"].strip(), r["intent"].strip()
            if not text:
                continue
            if intent not in LABELS:
                raise ValueError(f"Unknown intent {intent!r} in {CUSTOM_CSV}: {text}")
            rows.append({"text": text, "intent": intent, "source_intent": "custom",
                         "locale": "custom", "split": "train"})
    return rows


def downsample_other(rows, ratio, seed=42):
    """"other" is ~90% of MASSIVE; keep only `ratio` of it per split."""
    rng = random.Random(seed)
    return [r for r in rows if r["intent"] != "other" or rng.random() < ratio]


def load_intent_data(other_ratio=0.12, locales=("th-TH", "en-US"), seed=42):
    """Return {"train": [...], "dev": [...], "test": [...]} of row dicts."""
    rows = downsample_other(load_massive(locales), other_ratio, seed) + load_custom()
    out = {s: [] for s in SPLITS.values()}
    for r in rows:
        out[r["split"]].append(r)
    return out


_THAI = r"฀-๿"
_THAI_GAP = re.compile(rf"(?<=[{_THAI}])\s+(?=[{_THAI}])")


def normalize(text):
    """Lowercase and drop spaces *between Thai characters*.

    MASSIVE Thai is pre-segmented with spaces ("ปลุกฉัน ตอน ตีห้า"), but real
    typing / Whisper output is not, so we remove them and re-tokenize the same way
    at train and inference time.
    """
    text = text.lower().strip()
    return _THAI_GAP.sub("", text)


def tokenize(text):
    return [t for t in word_tokenize(normalize(text), engine="newmm", keep_whitespace=False) if t.strip()]


def unigrams_bigrams(tokens):
    """TF-IDF analyzer over pre-tokenized text (module-level so joblib can pickle it)."""
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
