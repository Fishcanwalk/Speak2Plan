"""Rule-based slot extraction for write commands — no training, just regexes.

    parse_add_task("เพิ่ม task ซื้อนมพรุ่งนี้")   -> ("ซื้อนม", date(tomorrow))
    match_tasks("ซื้อนมเสร็จแล้ว", open_tasks)    -> [(1.0, {"title": "ซื้อนม", ...})]

The intent classifier only says *what* the user wants (add_task / complete_task);
these rules pull out *which* task and *when*. Every write is confirmed by the user
afterwards, so a wrong guess here is cancelled rather than saved.
"""
import re
from datetime import date, timedelta
from difflib import SequenceMatcher

TH_WEEKDAYS = {"จันทร์": 0, "อังคาร": 1, "พุธ": 2, "พฤหัส": 3, "พฤหัสบดี": 3, "ศุกร์": 4, "เสาร์": 5, "อาทิตย์": 6}
EN_WEEKDAYS = {d: i for i, d in enumerate(
    ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])}


# ---------- due date ----------

def _next_weekday(today, weekday, next_week=False):
    """Coming `weekday` (today counts); `next_week` = that weekday in next calendar week
    ("จันทร์หน้า" / "next monday" said on a Tuesday = the Monday 6 days later, not 13)."""
    if next_week:
        return today + timedelta(7 - today.weekday() + weekday)
    return today + timedelta((weekday - today.weekday()) % 7)


def _day_of_month(today, day):
    """'วันที่ 10' = the 10th of this month, or next month if it has passed."""
    try:
        d = today.replace(day=day)
        if d < today:
            d = (today.replace(day=1) + timedelta(32)).replace(day=day)
        return d
    except ValueError:  # e.g. วันที่ 31 in a 30-day month
        return None


_TH_DIGIT = {"หนึ่ง": 1, "เอ็ด": 1, "สอง": 2, "ยี่": 2, "สาม": 3, "สี่": 4, "ห้า": 5, "หก": 6, "เจ็ด": 7,
            "แปด": 8, "เก้า": 9}
_TH_DAY_NUM = r"(\d{1,2}|(?:ยี่|สาม)?สิบ(?:เอ็ด|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)?|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า)"


def _th_int(s):
    """'15' / 'สิบห้า' / 'ยี่สิบเอ็ด' / 'สาม' -> int (1–39)."""
    if s.isdigit():
        return int(s)
    if "สิบ" not in s:
        return _TH_DIGIT[s]
    tens, _, ones = s.partition("สิบ")
    return (_TH_DIGIT[tens] if tens else 1) * 10 + (_TH_DIGIT[ones] if ones else 0)


def _next_month_day(today, day):
    first = (today.replace(day=1) + timedelta(32)).replace(day=1)
    try:
        return first.replace(day=day)
    except ValueError:
        return None


_PREFIX = r"(?:ภายใน|ก่อน|ครบกำหนด|กำหนดส่ง|(?<![a-z])(?:due|by|on)\s+)?\s*"
_DATE_RULES = [
    (_PREFIX + r"เดือนหน้า\s*(?:วันที่)?\s*" + _TH_DAY_NUM, lambda m, t: _next_month_day(t, _th_int(m[1]))),
    (_PREFIX + r"(?:วันนี้|today|tonight|คืนนี้|เย็นนี้|บ่ายนี้|เช้านี้)", lambda m, t: t),
    (_PREFIX + r"(?:พรุ่งนี้|tomorrow)", lambda m, t: t + timedelta(1)),
    (_PREFIX + r"มะรืน(?:นี้)?", lambda m, t: t + timedelta(2)),
    (_PREFIX + r"วันที่\s*" + _TH_DAY_NUM, lambda m, t: _day_of_month(t, _th_int(m[1]))),
    (_PREFIX + r"(?:วัน)?(จันทร์|อังคาร|พุธ|พฤหัส(?:บดี)?|ศุกร์|เสาร์|อาทิตย์)(หน้า|นี้)?",
     lambda m, t: _next_weekday(t, TH_WEEKDAYS[m[1]], m[2] == "หน้า")),
    (_PREFIX + r"(next |this )?(" + "|".join(EN_WEEKDAYS) + r")",
     lambda m, t: _next_weekday(t, EN_WEEKDAYS[m[2]], m[1] == "next ")),
]
_DATE_RULES = [(re.compile(p, re.I), fn) for p, fn in _DATE_RULES]


def extract_date(text, today=None):
    """Return (date or None, text with the date phrase removed)."""
    today = today or date.today()
    for rx, fn in _DATE_RULES:
        m = rx.search(text)
        if m:
            return fn(m, today), (text[:m.start()] + " " + text[m.end():]).strip()
    return None, text


# ---------- task title ----------

# Peeled off the front in this order: politeness, verb, "task"-noun, joiners.
_LEAD = [
    (r"(?:ช่วย|ขอ|อย่าลืม|please)\s*", True),
    (r"(?:เพิ่ม|ใส่|สร้าง|จด(?:ไว้)?|ตั้ง|add|new|put|create|remind me to|เตือนให้|เตือน)\s*", False),
    (r"(?:ลง)?(?:ใน)?\s*(?:a\s+)?(?:google\s*)?(?:tasks?|ทาสก์|งาน|to ?do|ลิสต์|list)\s*", False),
    (r"(?:ใหม่|หนึ่งอัน|ไว้|หน่อย|ให้หน่อย|ให้(?=ว่า)|ใน task|ที่ชื่อ|ชื่อ|ว่า(?:ต้อง)?|:|to\s+)\s*", True),
]
_LEAD = [(re.compile("^" + p, re.I), repeat) for p, repeat in _LEAD]
# Peeled off the end repeatedly: "...ลงใน to do", "...to my tasks", "...ให้หน่อย"
_TRAIL = re.compile(
    r"\s*(?:(?:ลง|เข้า)?(?:ไป)?(?:ใน|to|on|in)?\s*(?:my\s+)?(?:task|tasks|ทาสก์|to ?do(?:\s*list)?|ลิสต์|list)"
    r"|ลงไป|เข้าไป|ให้หน่อย|ให้ที|หน่อย|ด้วย|นะ|ครับ|คับ|ค่ะ|คะ)$", re.I)


def extract_title(text):
    t = text.strip()
    for rx, repeat in _LEAD:
        while (m := rx.match(t)) and m.end():
            t = t[m.end():]
            if not repeat:
                break
    while (m := _TRAIL.search(t)) and m.start() < len(t):
        t = t[:m.start()]
    return " ".join(t.strip(" .,!?").split())


def parse_add_task(text, today=None):
    """'เพิ่ม task ซื้อนมพรุ่งนี้' -> ('ซื้อนม', tomorrow). Title may be '' if nothing is left."""
    due, rest = extract_date(text, today)
    return extract_title(rest), due


# ---------- event time ----------

_TH_NUM = {"หนึ่ง": 1, "นึง": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5, "หก": 6, "เจ็ด": 7, "แปด": 8,
           "เก้า": 9, "สิบ": 10, "สิบเอ็ด": 11, "สิบสอง": 12}
_EN_NUM = {w: i for i, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve".split(), start=1)}
_N = r"(\d{1,2}|" + "|".join(sorted(_TH_NUM, key=len, reverse=True)) + ")"
_EN = r"(\d{1,2}|" + "|".join(_EN_NUM) + ")"


def _num(s):
    return int(s) if s.isdigit() else _TH_NUM.get(s) or _EN_NUM.get(s.lower())


def _half(m):
    return 30 if m.group("half") else 0


_HALF = r"\s*(?P<half>ครึ่ง)?"
# Thai clock: บ่าย = afternoon, โมงเย็น = evening, ทุ่ม = 7–11 pm, ตี = 1–5 am. First match wins.
_EN_MIN = {"fifteen": 15, "thirty": 30, "forty five": 45, "forty-five": 45, "o'clock": 0, "oclock": 0}
_EN_MIN_RE = r"(?:\s+(" + "|".join(_EN_MIN) + r"))?"


def _en_hour(h):
    """English "at four" / "at 4:30" without am/pm: 1–6 are afternoon (nobody books 4 am)."""
    return h + 12 if 1 <= h <= 6 else h


_TIME_RULES = [
    (r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)",
     lambda m: (int(m[1]) % 12 + (12 if m[3].startswith("p") else 0), int(m[2] or 0))),
    (r"\bat\s+(\d{1,2})[:.](\d{2})\b", lambda m: (_en_hour(int(m[1])), int(m[2]))),
    (r"(\d{1,2})[:.](\d{2})\s*(?:น\.?|นาฬิกา)?", lambda m: (int(m[1]), int(m[2]))),
    (r"เที่ยงคืน", lambda m: (0, 0)),
    (r"เที่ยง(?:วัน|ตรง)?" + _HALF, lambda m: (12, _half(m))),
    (r"บ่าย\s*(?:" + _N + r")?\s*(?:โมง)?" + _HALF, lambda m: (12 + (_num(m[1]) if m[1] else 1), _half(m))),
    (_N + r"\s*โมง\s*เย็น" + _HALF, lambda m: (12 + _num(m[1]) % 12, _half(m))),
    (r"(?:" + _N + r"\s*ทุ่ม|ทุ่ม\s*(" + _N[1:-1] + r")?)" + _HALF,
     lambda m: (18 + (_num(m[1] or m[2]) if (m[1] or m[2]) else 1), _half(m))),
    (r"ตี\s*" + _N + _HALF, lambda m: (_num(m[1]), _half(m))),
    # "N โมง(เช้า)": 7–11 = morning; 1–6 without เช้า = afternoon (Thai speakers say บ่าย/เย็น loosely)
    (_N + r"\s*โมง\s*(เช้า)?" + _HALF,
     lambda m: (_num(m[1]) + (12 if _num(m[1]) <= 6 and not m[2] else 0), _half(m))),
    (r"\bnoon\b", lambda m: (12, 0)),
    (r"\bmidnight\b", lambda m: (0, 0)),
    (r"\bat\s+" + _EN + _EN_MIN_RE + r"\b",
     lambda m: (_en_hour(_num(m[1])), _EN_MIN[m[2].lower()] if m[2] else 0)),
]
_TIME_RULES = [(re.compile(p, re.I), fn) for p, fn in _TIME_RULES]


def extract_time(text):
    """Return ((hour, minute) or None, text with the time phrase removed)."""
    for rx, fn in _TIME_RULES:
        m = rx.search(text)
        if m:
            h, mi = fn(m)
            if 0 <= h <= 23 and 0 <= mi <= 59:
                return (h, mi), (text[:m.start()] + " " + text[m.end():]).strip()
    return None, text


# Command words peeled off the front (any order, repeatedly): "ช่วย ลง calendar ว่า มี นัด ..."
_EVENT_LEAD = re.compile(
    r"^(?:ช่วย|ขอ|please|เพิ่ม|ใส่|สร้าง|ลง|ตั้ง|จอง(?:เวลา)?|บันทึก|add|create|schedule|book|put"
    r"|an?|my|new|event|นัด(?:หมาย)?|ปฏิทิน|calendar|ตาราง|ใหม่|ใน|to|ว่า|มี|ไว้)\s*", re.I)
_EVENT_TRAIL = re.compile(
    r"\s*(?:(?:ลง|ใส่|ใน|to|on|in)?\s*(?:my\s+)?(?:calendar|ปฏิทิน|ตาราง)(?:ให้)?"
    r"|ให้หน่อย|ให้ที|หน่อย|ด้วย|ไว้|นะ|ครับ|คับ|ค่ะ|คะ|ตอน|เวลา|at|on|วัน|ช่วย)$", re.I)
# Part-of-day / repetition words left behind once the date and time are taken out
_EVENT_FILLER = re.compile(
    r"(?:^|\s)(?:ตอน)?(?:เช้า|สาย|บ่าย|เย็น|คืน|กลางคืน)(?=\s|$)|(?<=\S)(?:ตอน)?(?:เช้า|เย็น|คืน)(?=\s|$)"
    r"|ทุก(?=\s|$)|\b(?:every|morning|afternoon|evening|night)\b", re.I)


def parse_add_event(text, today=None):
    """'นัดหมอฟันวันจันทร์บ่ายสอง' -> ('หมอฟัน', date(next Mon), (14, 0)).

    Returns (title, date or None, (hour, minute) or None). A time with no date means today
    (tomorrow if that time has passed — the caller decides, since it knows the clock).
    """
    when, rest = extract_date(text, today)
    clock, rest = extract_time(rest)
    if when is None:  # "เดือนหน้า" etc. are not handled; time-only stays dateless
        when, rest = extract_date(rest, today)
    title = rest.strip()
    while (m := _EVENT_LEAD.match(title)) and m.end():
        title = title[m.end():]
    while (m := _EVENT_TRAIL.search(title)) and m.start() < len(title):
        title = title[:m.start()]
    title = _EVENT_FILLER.sub(" ", title)
    title = re.sub(r"^(?:ไว้|ว่า)\s*", "", title.strip())
    title = re.sub(r"\s+(?:วัน|ตอน|เวลา|at|on)\s+", " ", f" {title} ").strip(" .,!?")
    return " ".join(title.split()), when, clock


# ---------- which existing task ----------

def _norm(s):
    return re.sub(r"[\s.,!?'\"-]", "", s.lower())


def match_tasks(text, tasks, min_score=0.45, min_chars=4):
    """Rank open tasks by how much of their title appears (in order) in `text`.

    Fuzzy on purpose: ASR rarely spells a title exactly, and people say only part of it
    ("จองโรงแรม" for "จองโรงแรมที่ภูเก็ต"). Blocks shorter than 2 chars are ignored so scattered
    single letters don't count; a match needs >= `min_chars` matched characters.
    Returns [(score, task)], best first.
    """
    t = _norm(text)
    ranked = []
    for task in tasks:
        title = _norm(task["title"])
        if not title:
            continue
        blocks = SequenceMatcher(None, title, t, autojunk=False).get_matching_blocks()
        matched = sum(b.size for b in blocks if b.size >= 2)
        score = matched / len(title)
        if score >= min_score and matched >= min(min_chars, len(title)):
            ranked.append((score, task))
    return sorted(ranked, key=lambda x: -x[0])
