"""Record your own voice commands as an ASR + intent test set (default) or train set.

    python -m src.record                 # test set:  data/audio/
    python -m src.record --set train     # train set: data/audio_train/ (different sentences!)
    python -m src.record --redo cmd05    # re-record one file

Reads prompts from <dir>/prompts.csv (text,intent) — edit it to say things your way.
For each prompt: Enter = start, Enter = stop, then [Enter] keep / r re-record / p play / s skip / q quit.
Saves <dir>/cmdNN.wav (16 kHz mono) and appends to <dir>/transcripts.csv (file,text,intent).
The test set is read by src.eval_e2e / src.asr eval; the train set by src.train_asr (--own-repeat)
and src.train_intent. Never record test-set sentences into the train set.
Uses `arecord` (alsa-utils) with the system default microphone.
"""
import argparse
import csv
import subprocess
from pathlib import Path

from .data import DATA_DIR

SET_DIRS = {"test": DATA_DIR / "audio", "train": DATA_DIR / "audio_train"}
FIELDS = ["file", "text", "intent"]


def read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_transcripts(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
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
    p.add_argument("--set", choices=SET_DIRS, default="test")
    p.add_argument("--redo", help="file stem to re-record, e.g. cmd05")
    args = p.parse_args()

    audio_dir = SET_DIRS[args.set]
    transcripts = audio_dir / "transcripts.csv"
    audio_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_csv(audio_dir / "prompts.csv")
    if not prompts:
        raise SystemExit(f"no prompts in {audio_dir / 'prompts.csv'}")
    done = read_csv(transcripts)
    if args.set == "train":   # guard against test-set leakage
        test_dir = SET_DIRS["test"]
        test_texts = {r["text"] for r in read_csv(test_dir / "prompts.csv") + read_csv(test_dir / "transcripts.csv")}
        leaked = [r["text"] for r in prompts if r["text"] in test_texts]
        if leaked:
            raise SystemExit(f"these train prompts are in the test set, remove them: {leaked}")

    if args.redo:
        row = next((r for r in done if Path(r["file"]).stem == args.redo), None)
        if row is None:
            raise SystemExit(f"{args.redo} not in {transcripts}")
        record_one(audio_dir / row["file"], row["text"])
        return

    recorded = {r["text"] for r in done}
    todo = [r for r in prompts if r["text"] not in recorded]
    print(f"{len(done)} recorded, {len(todo)} to go. พูดเป็นธรรมชาติ ไม่ต้องอ่านเป๊ะทุกคำ "
          "(ถ้าพูดต่างจากประโยค ให้แก้ text ใน transcripts.csv ทีหลัง)")
    n = max([int(Path(r["file"]).stem[3:]) for r in done] + [0])
    for prompt in todo:
        n += 1
        path = audio_dir / f"cmd{n:02d}.wav"
        kept = record_one(path, prompt["text"])
        if kept is None:
            break
        if kept:
            done.append({"file": path.name, "text": prompt["text"], "intent": prompt["intent"]})
            write_transcripts(transcripts, done)
        else:
            n -= 1
    print(f"\n{len(done)} recordings in {transcripts.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
