"""Record your own voice commands as an ASR + intent test set.

    python -m src.record                 # record every prompt not yet recorded
    python -m src.record --redo cmd05    # re-record one file

Reads prompts from data/audio/prompts.csv (text,intent) — edit it to say things your way.
For each prompt: Enter = start, Enter = stop, then [Enter] keep / r re-record / p play / s skip / q quit.
Saves data/audio/cmdNN.wav (16 kHz mono) and appends to data/audio/transcripts.csv (file,text,intent),
which `python -m src.asr eval --data data/audio/transcripts.csv` reads.
Uses `arecord` (alsa-utils) with the system default microphone.
"""
import argparse
import csv
import subprocess
from pathlib import Path

from .data import DATA_DIR

AUDIO_DIR = DATA_DIR / "audio"
PROMPTS = AUDIO_DIR / "prompts.csv"
TRANSCRIPTS = AUDIO_DIR / "transcripts.csv"
FIELDS = ["file", "text", "intent"]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_transcripts(rows):
    with open(TRANSCRIPTS, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def record(path):
    input("  [Enter] เริ่มอัด ... ")
    proc = subprocess.Popen(["arecord", "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", str(path)])
    input("  ● กำลังอัด — พูดเลย แล้วกด [Enter] เพื่อหยุด ")
    proc.terminate()
    proc.wait()


def record_one(path, text):
    """Record until the user keeps it. Returns True if kept, False if skipped, None to quit."""
    print(f"\n{path.name}:  「{text}」")
    while True:
        record(path)
        while True:
            ans = input("  [Enter] เก็บ / r อัดใหม่ / p ฟัง / s ข้าม / q ออก: ").strip().lower()
            if ans == "p":
                subprocess.run(["aplay", "-q", str(path)])
                continue
            break
        if ans == "":
            return True
        if ans in ("s", "q"):
            path.unlink(missing_ok=True)
            return False if ans == "s" else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--redo", help="file stem to re-record, e.g. cmd05")
    args = p.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    prompts = read_csv(PROMPTS)
    done = read_csv(TRANSCRIPTS)

    if args.redo:
        row = next((r for r in done if Path(r["file"]).stem == args.redo), None)
        if row is None:
            raise SystemExit(f"{args.redo} not in {TRANSCRIPTS}")
        record_one(AUDIO_DIR / row["file"], row["text"])
        return

    recorded = {r["text"] for r in done}
    todo = [r for r in prompts if r["text"] not in recorded]
    print(f"{len(done)} recorded, {len(todo)} to go. พูดเป็นธรรมชาติ ไม่ต้องอ่านเป๊ะทุกคำ "
          "(ถ้าพูดต่างจากประโยค ให้แก้ text ใน transcripts.csv ทีหลัง)")
    n = max([int(Path(r["file"]).stem[3:]) for r in done] + [0])
    for prompt in todo:
        n += 1
        path = AUDIO_DIR / f"cmd{n:02d}.wav"
        kept = record_one(path, prompt["text"])
        if kept is None:
            break
        if kept:
            done.append({"file": path.name, "text": prompt["text"], "intent": prompt["intent"]})
            write_transcripts(done)
        else:
            n -= 1
    print(f"\n{len(done)} recordings in {TRANSCRIPTS.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
