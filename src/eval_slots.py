"""Slot accuracy for write commands: is the task/event that would be saved actually right?

    python -m src.eval_e2e                       # first: makes reports/e2e/utterances.csv
    python -m src.eval_slots
    python -m src.eval_slots --asr whisper-small-th --intent lstm

Gold answers: data/slots_gold.csv (text,intent,title,day,time). `day` is relative to --today:
"+1" = tomorrow, "sat" = the coming Saturday (today counts), "next:mon" = Monday of next week.
complete_task is matched against a simulated open-task list (the add_task titles + distractors).

A command "succeeds" when the intent is right AND title (similarity >= 0.8), date and time are right.
Outputs reports/e2e/slots.json.
"""
import argparse
import csv
import json
from datetime import date, timedelta
from difflib import SequenceMatcher

from .data import DATA_DIR, ROOT
from .slots import EN_WEEKDAYS, match_tasks, parse_add_event, parse_add_task

GOLD_CSV = DATA_DIR / "slots_gold.csv"
REPORT_DIR = ROOT / "reports" / "e2e"
DISTRACTORS = ["Cert 3", "Cyber Security Cert", "ส่งรายงาน", "ซื้อนม", "อ่านหนังสือ"]
WD = {d[:3]: i for d, i in EN_WEEKDAYS.items()}


def gold_day(spec, today):
    if not spec:
        return None
    if spec.startswith("+"):
        return today + timedelta(int(spec[1:]))
    if spec.startswith("next:"):
        monday = today + timedelta(7 - today.weekday())
        return monday + timedelta(WD[spec[5:]])
    return today + timedelta((WD[spec] - today.weekday()) % 7)


def similar(a, b):
    a, b = ("".join(x.lower().split()) for x in (a, b))
    return SequenceMatcher(None, a, b).ratio()


def predict_slots(intent, text, today, open_tasks):
    """(title, day, "HH:MM" or None) the pipeline would use for this text."""
    if intent == "add_task":
        title, due = parse_add_task(text, today)
        return title, due, None
    if intent == "add_event":
        title, day, clock = parse_add_event(text, today)
        return title, day, f"{clock[0]:02d}:{clock[1]:02d}" if clock else None
    if intent == "complete_task":
        ranked = match_tasks(text, open_tasks)
        unique = ranked and (len(ranked) == 1 or ranked[1][0] < ranked[0][0])
        return (ranked[0][1]["title"] if unique else ""), None, None
    return "", None, None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--utterances", default=str(REPORT_DIR / "utterances.csv"))
    p.add_argument("--asr", default="whisper-small-th-v2", help="source column name in utterances.csv")
    p.add_argument("--intent", default="nb")
    p.add_argument("--today", default="2026-10-06", help="date the relative gold days are counted from")
    args = p.parse_args()
    today = date.fromisoformat(args.today)

    with open(GOLD_CSV, encoding="utf-8") as f:
        gold = {r["text"]: r for r in csv.DictReader(f)}
    open_tasks = [{"id": str(i), "title": t} for i, t in enumerate(
        [g["title"] for g in gold.values() if g["intent"] == "add_task"] + DISTRACTORS)]

    with open(args.utterances, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ref_text = {r["file"]: r["text"] for r in rows if r["source"] == "reference"}

    results, detail = {}, []
    for source in ["reference", args.asr]:
        per = {}
        for r in rows:
            if r["source"] != source or ref_text[r["file"]] not in gold:
                continue
            g = gold[ref_text[r["file"]]]
            pred_intent = r[f"pred_{args.intent}"]
            title, day, clock = predict_slots(g["intent"], r["text"], today, open_tasks)
            ok = {
                "intent": pred_intent == g["intent"],
                "title": similar(title, g["title"]) >= 0.8,
                "day": day == gold_day(g["day"], today),
                "time": clock == (g["time"] or None),
            }
            ok["all_slots"] = ok["title"] and ok["day"] and ok["time"]
            ok["success"] = ok["intent"] and ok["all_slots"]
            per.setdefault(g["intent"], []).append(ok)
            detail.append({"source": source, "file": r["file"], "intent": g["intent"], "text": r["text"],
                           "pred_intent": pred_intent, "title": title, "day": str(day or ""),
                           "time": clock or "", **{f"ok_{k}": v for k, v in ok.items()}})
        results[source] = {
            intent: {k: sum(o[k] for o in oks) / len(oks) for k in oks[0]} | {"n": len(oks)}
            for intent, oks in per.items()}
        all_oks = [o for oks in per.values() for o in oks]
        results[source]["all"] = {k: sum(o[k] for o in all_oks) / len(all_oks) for k in all_oks[0]} | {
            "n": len(all_oks)}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "slots.json").write_text(json.dumps(
        {"asr": args.asr, "intent": args.intent, "today": args.today, "results": results, "detail": detail},
        ensure_ascii=False, indent=2))

    cols = ["intent", "title", "day", "time", "all_slots", "success"]
    for source, res in results.items():
        print(f"\n== text from {source}  (intent model: {args.intent})")
        print(f"  {'':15s}" + "".join(f"{c:>10s}" for c in cols) + "      n")
        for intent, m in res.items():
            print(f"  {intent:15s}" + "".join(f"{m[c]:10.2f}" for c in cols) + f"  {m['n']:5d}")
    print("\nwrong slots:")
    for d in detail:
        if not d["ok_all_slots"]:
            print(f"  [{d['source']}] {d['file']} {d['intent']}: '{d['text']}' -> "
                  f"title='{d['title']}' day={d['day'] or '-'} time={d['time'] or '-'}"
                  f"  ({', '.join(k for k in ('title', 'day', 'time') if not d['ok_' + k])} wrong)")
    print("\nsaved reports/e2e/slots.json")


if __name__ == "__main__":
    main()
