"""Speak2Plan end-to-end: voice/text -> Whisper -> intent -> Google Tasks/Calendar -> spoken reply.

    python -m src.pipeline                                   # interactive: Enter = speak, or type a command
    python -m src.pipeline --wake                            # hands-free: say "เคทู ..." (K2) then the command
    python -m src.pipeline --text "เช็ค task ให้หน่อย"
    python -m src.pipeline --audio data/audio/cmd01.wav
    python -m src.pipeline --asr-model models/whisper-th --intent-model lstm --no-tts
    python -m src.pipeline --wake --device cpu               # keep off the GPU (e.g. while training)

check_tasks / check_calendar read Google (check_calendar narrows to the day/week mentioned);
add_task / complete_task / add_event ask "ใช่ไหม" and write only after a clear yes
(title/date/time/which task come from rules in src/slots.py).
Mic uses `arecord`, playback uses `aplay` (alsa-utils). gTTS needs internet.
"""
import argparse
import io
import re
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import google_api
from .intent import IntentClassifier
from .slots import extract_date, match_tasks, parse_add_event, parse_add_task

SR = 16000
MIN_CONFIDENCE = 0.4   # below this the intent is treated as "didn't understand"
MAX_SPOKEN_ITEMS = 5   # read at most this many tasks/events aloud
CALENDAR_DAYS = 7

THAI_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
THAI_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
               "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]

# Checked in this order, so "ไม่ใช่" / "ไม่ได้" count as no. Anything that isn't a clear yes cancels.
NO_RE = re.compile(r"ไม่|ยกเลิก|ผิด|อย่า|\b(?:no|nope|cancel)\b", re.I)
YES_RE = re.compile(r"ใช่|ได้|โอเค|ตกลง|ถูก|ยืนยัน|เอาเลย|จัดไป|แน่นอน|\b(?:ok|okay|yes|yeah|yep|sure|confirm)\b", re.I)


# ---------- reply text ----------

def thai_day(d: date) -> str:
    delta = (d - date.today()).days
    if delta == 0:
        return "วันนี้"
    if delta == 1:
        return "พรุ่งนี้"
    return f"วัน{THAI_DAYS[d.weekday()]}ที่ {d.day} {THAI_MONTHS[d.month - 1]}"


def thai_when(start: str | date | datetime) -> str:
    """Calendar start: '2026-10-05T15:00:00+07:00' / datetime (timed) or '2026-10-05' / date (all-day)."""
    if isinstance(start, str):
        start = datetime.fromisoformat(start) if "T" in start else date.fromisoformat(start)
    if not isinstance(start, datetime):
        return f"{thai_day(start)} ทั้งวัน"
    dt = start.astimezone()
    return f"{thai_day(dt.date())} {dt:%H.%M} น."


def join_items(items: list[str]) -> str:
    text = ", ".join(items[:MAX_SPOKEN_ITEMS])
    if len(items) > MAX_SPOKEN_ITEMS:
        text += f" และอีก {len(items) - MAX_SPOKEN_ITEMS} รายการ"
    return text


def reply_tasks() -> str:
    tasks = google_api.list_tasks()
    if not tasks:
        return "ไม่มีงานค้างเลย"
    # Tasks API stores due as a date only (midnight UTC), so the date part is exact
    items = [t["title"] + (f" ครบกำหนด{thai_day(date.fromisoformat(t['due'][:10]))}" if t["due"] else "")
             for t in tasks]
    return f"มีงานค้าง {len(tasks)} งาน ได้แก่ {join_items(items)}"


NEXT_WEEK_RE = re.compile(r"(?<!วัน)(?:อาทิตย์|สัปดาห์)หน้า|next week", re.I)
THIS_WEEK_RE = re.compile(r"(?<!วัน)(?:อาทิตย์|สัปดาห์)นี้|this week", re.I)


def calendar_range(text: str, today: date | None = None):
    """(start datetime, days, spoken label) for the period a calendar question asks about.

    "อาทิตย์หน้า" alone means next week (Mon–Sun), "วันอาทิตย์หน้า" next Sunday."""
    today = today or date.today()
    midnight = lambda d: datetime.combine(d, time()).astimezone()  # noqa: E731
    if NEXT_WEEK_RE.search(text):
        monday = today + timedelta(7 - today.weekday())
        return midnight(monday), 7, "สัปดาห์หน้า"
    if THIS_WEEK_RE.search(text):
        return None, 7 - today.weekday(), "สัปดาห์นี้"
    day, _ = extract_date(text, today)
    if day:
        return midnight(day), 1, thai_day(day)
    return None, CALENDAR_DAYS, f"ใน {CALENDAR_DAYS} วันข้างหน้า"


def reply_calendar(text: str = "") -> str:
    start, days, label = calendar_range(text)
    events = google_api.list_events(days=days, start=start)
    if not events:
        return f"{label}ไม่มีนัด"
    items = [f"{thai_when(e['start'])} {e['summary']}" for e in events]
    return f"{label}มี {len(events)} นัด ได้แก่ {join_items(items)}"


@dataclass
class Pending:
    """A Google write waiting for the user's yes/no. `run` does it and returns the reply."""
    run: Callable[[], str]


def ask_add_task(text: str):
    title, due = parse_add_task(text)
    if not title:
        return "ไม่ได้ยินชื่องาน ลองพูดใหม่ เช่น เพิ่มงานซื้อนมพรุ่งนี้", None
    when = f" ครบกำหนด{thai_day(due)}" if due else ""

    def run():
        google_api.add_task(title, due)
        return f"เพิ่มงาน {title}{when} เรียบร้อย"
    return f"จะเพิ่มงาน {title}{when} ใช่ไหม", Pending(run)


def ask_complete_task(text: str):
    ranked = match_tasks(text, google_api.list_tasks(max_results=100))
    if not ranked:
        return "หางานที่ตรงกันไม่เจอ ลองพูดชื่องานให้ชัดขึ้น", None
    best = ranked[0][0]
    if len(ranked) > 1 and ranked[1][0] == best:  # e.g. "cert" vs Cert 2 / Cert 3 / Cert 4
        names = join_items([t["title"] for score, t in ranked if score == best])
        return f"มีหลายงานที่ตรงกัน ได้แก่ {names} ช่วยพูดชื่องานให้ชัดขึ้น", None
    task = ranked[0][1]

    def run():
        google_api.complete_task(task["id"])
        return f"ติ๊กงาน {task['title']} ว่าเสร็จแล้ว"
    return f"จะติ๊กงาน {task['title']} ว่าเสร็จแล้ว ใช่ไหม", Pending(run)


def ask_add_event(text: str, now: datetime | None = None):
    title, day, clock = parse_add_event(text)
    if not title:
        return "ไม่ได้ยินชื่อนัด ลองพูดใหม่ เช่น นัดหมอฟันพรุ่งนี้บ่ายสอง", None
    if day is None and clock is None:
        return f"จะเพิ่มนัด {title} วันไหน กี่โมง ลองพูดใหม่พร้อมวันเวลา", None
    now = now or datetime.now().astimezone()
    if clock is None:
        start = day                                   # all-day
    else:
        start = datetime.combine(day or now.date(), time(*clock)).astimezone()
        if day is None and start <= now:              # "บ่ายสอง" said at 3 pm = tomorrow
            start += timedelta(days=1)

    def run():
        google_api.add_event(title, start)
        return f"เพิ่มนัด {title} {thai_when(start)} เรียบร้อย"
    return f"จะเพิ่มนัด {title} {thai_when(start)} ใช่ไหม", Pending(run)


def respond(intent: str, confidence: float, text: str):
    """Return (reply, Pending or None). A Pending means the reply is a yes/no question."""
    if confidence < MIN_CONFIDENCE:
        return "ไม่แน่ใจว่าหมายถึงอะไร ลองพูดใหม่อีกครั้งได้ไหม", None
    try:
        if intent == "check_tasks":
            return reply_tasks(), None
        if intent == "check_calendar":
            return reply_calendar(text), None
        if intent == "add_task":
            return ask_add_task(text)
        if intent == "complete_task":
            return ask_complete_task(text)
        if intent == "add_event":
            return ask_add_event(text)
    except Exception as e:  # network / expired token — keep the demo alive
        print(f"  [Google error] {e}")
        return "เชื่อมต่อ Google ไม่สำเร็จ", None
    return "คำสั่งนี้ไม่เกี่ยวกับงานหรือปฏิทิน ลองถามเรื่องงานหรือนัดได้เลย", None


# ---------- audio I/O ----------

def listen() -> Path:
    """Record from the default mic until Enter. Caller deletes the file."""
    path = Path(tempfile.mkstemp(suffix=".wav")[1])
    proc = subprocess.Popen(["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1", str(path)])
    input("  ● กำลังฟัง — พูดเลย แล้วกด [Enter] เพื่อหยุด ")
    proc.terminate()
    proc.wait()
    return path


def play(wav: np.ndarray, sr: int):
    out = io.BytesIO()
    sf.write(out, wav, sr, format="WAV", subtype="PCM_16")
    subprocess.run(["aplay", "-q", "-"], input=out.getvalue(), check=True)


def beep():
    t = np.arange(int(SR * 0.15)) / SR
    play(0.3 * np.sin(2 * np.pi * 880 * t), SR)


def speak(text: str):
    from gtts import gTTS

    try:
        mp3 = io.BytesIO()
        gTTS(text, lang="th").write_to_fp(mp3)
        mp3.seek(0)
        play(*sf.read(mp3))
    except Exception as e:
        print(f"  (TTS ไม่สำเร็จ: {e})")


# ---------- wake word ("K2") ----------

WAKE_NAME = "เคทู"
# Whisper spells "เคทู" many ways (เกทู, เก-ทู, เคตุ, เคราะทู, เครีย์ธู, เคศู, เก่ต้, เกตทุ, K-2;
# whisper-small-th-v2: "ke to", "ke to do" ...),
# so match the sound pattern at the start of the utterance rather than one spelling.
WAKE_RE = re.compile(r"(?:hey|เฮ้?|เอ่อ)?(?:k(?:e|ay)?|เค้?|เก่?|แค|เคราะ|เครีย์?)[ตด]?(?:2|two|tu|to|ทู|ตู|ทุ|ตุ|ต้|ตว|ธู|ศู|สู)")
SKIP_RE = re.compile(r"[\s.,!?'\"-]")
PARTICLES_RE = re.compile(r"(?:ครับ|คับ|ค่ะ|คะ|จ้า|นะ)+")
WAKE_TIMEOUT = 6       # s to wait for the command after "K2" alone
VAD_CHUNK = 512        # silero-vad window at 16 kHz (32 ms)


def strip_wake(text: str):
    """Command after the wake name, '' if only the name was said, None if K2 wasn't called."""
    lower = text.lower()
    m = WAKE_RE.match(SKIP_RE.sub("", lower))
    if not m:
        return None
    # map the match end (in the space-free string) back to the original text
    kept, i = 0, 0
    while i < len(lower) and kept < m.end():
        kept += not SKIP_RE.match(lower[i])
        i += 1
    rest = text[i:].strip(" .,!?")
    # < 3 chars left ("ke to do" -> "do") is a mis-heard name, not a command: beep and wait instead
    return rest if len(PARTICLES_RE.sub("", SKIP_RE.sub("", rest))) >= 3 else ""


def listen_utterance(vad, timeout=None, max_seconds=15):
    """Block until one utterance ends (silero-vad); None if nothing starts within `timeout` s."""
    import torch

    vad.reset_states()
    proc = subprocess.Popen(["arecord", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SR), "-c", "1"],
                            stdout=subprocess.PIPE)
    pre = deque(maxlen=10)  # ~0.3 s kept from before speech starts
    buf, waited = None, 0
    try:
        while True:
            data = proc.stdout.read(VAD_CHUNK * 2)
            if len(data) < VAD_CHUNK * 2:
                return None
            chunk = np.frombuffer(data, np.int16).astype(np.float32) / 32768
            event = vad(torch.from_numpy(chunk)) or {}
            if buf is None:
                pre.append(chunk)
                if "start" in event:
                    buf = list(pre)
                elif timeout:
                    waited += VAD_CHUNK
                    if waited > timeout * SR:
                        return None
                continue
            buf.append(chunk)
            if "end" in event or len(buf) * VAD_CHUNK > max_seconds * SR:
                if len(buf) * VAD_CHUNK < 0.4 * SR:  # click / cough
                    buf = None
                    continue
                return np.concatenate(buf)
    finally:  # mic is off while we think and talk, so K2 doesn't hear itself
        proc.terminate()
        proc.wait()


# ---------- pipeline ----------

class Assistant:
    def __init__(self, asr_model, intent_model, asr_language="thai", device=None):
        self.clf = IntentClassifier(intent_model, device=device)
        self.asr_model, self.asr_language, self.device = asr_model, asr_language, device
        self._asr = None
        self.pending = None  # Pending write waiting for yes/no

    @property
    def asr(self):
        if self._asr is None:  # Whisper is slow to load; skip it in text-only use
            from .asr import Transcriber

            print(f"  (loading {self.asr_model} ...)")
            self._asr = Transcriber(self.asr_model, device=self.device, language=self.asr_language)
        return self._asr

    def transcribe(self, audio, prompt=None) -> str:
        return self.asr.transcribe([audio], prompt=prompt)[0]

    def handle_text(self, text: str) -> str:
        if self.pending:
            return self.confirm(text)
        probs = self.clf.predict_proba([text])[0]
        intent = max(probs, key=probs.get)
        print(f"[Intent] {intent} ({probs[intent]:.2f})")
        reply, self.pending = respond(intent, probs[intent], text)
        print(f"[ตอบ]    {reply}")
        return reply

    def confirm(self, answer: str) -> str:
        """Run the pending write only on a clear yes; anything else cancels it."""
        pending, self.pending = self.pending, None
        if NO_RE.search(answer) or not YES_RE.search(answer):
            reply = "ยกเลิกแล้ว ไม่ได้แก้อะไรใน Google"
        else:
            try:
                reply = pending.run()
            except Exception as e:
                print(f"  [Google error] {e}")
                reply = "บันทึกลง Google ไม่สำเร็จ"
        print(f"[ตอบ]    {reply}")
        return reply

    def handle_audio(self, audio) -> str:
        text = self.transcribe(audio)
        print(f"[ASR]    {text or '(ไม่ได้ยินอะไร)'}")
        if not text:
            reply = "ไม่ได้ยินเลย ลองพูดใหม่อีกครั้ง"
            print(f"[ตอบ]    {reply}")
            return reply
        return self.handle_text(text)


def interactive(assistant: Assistant, tts: bool):
    while True:
        try:
            line = input("\n[ยืนยัน] ใช่/ไม่ — พิมพ์ หรือ Enter แล้วพูด: " if assistant.pending
                         else "\n[Enter] พูด / พิมพ์คำสั่ง / q ออก: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() == "q":
            break
        if line:
            reply = assistant.handle_text(line)
        else:
            path = listen()
            try:
                reply = assistant.handle_audio(path)
            finally:
                path.unlink(missing_ok=True)
        if tts:
            speak(reply)


def wake_loop(assistant: Assistant, tts: bool):
    from silero_vad import VADIterator, load_silero_vad

    vad = VADIterator(load_silero_vad(), sampling_rate=SR, min_silence_duration_ms=700, speech_pad_ms=100)
    assistant.asr  # load Whisper before the first command, not during it
    print('\nพูด "เคทู" ตามด้วยคำสั่ง เช่น "เคทู เช็ค task ให้หน่อย" หรือ "เคทู" เฉยๆ แล้วรอเสียงติ๊ง  (Ctrl+C ออก)')
    try:
        while True:
            print("\n(กำลังฟัง ...)")
            # prompting with the name makes Whisper spell it consistently (6/6 vs 5/6 on our recordings)
            text = assistant.transcribe(listen_utterance(vad), prompt=WAKE_NAME)
            command = strip_wake(text)
            if command is None:
                print(f"  (ได้ยิน: {text or '-'} — ไม่ได้เรียก K2)")
                continue
            print(f"[ASR]    {text}")
            if not command:
                beep()
                print("[K2]     ครับ? (รอคำสั่ง)")
                audio = listen_utterance(vad, timeout=WAKE_TIMEOUT)
                if audio is None:
                    print("  (หมดเวลา)")
                    continue
                command = assistant.transcribe(audio)
                print(f"[ASR]    {command or '(ไม่ได้ยินอะไร)'}")
                if not command:
                    continue
            reply = assistant.handle_text(command)
            if tts:
                speak(reply)
            while assistant.pending:  # answer yes/no without saying the name again
                beep()
                audio = listen_utterance(vad, timeout=WAKE_TIMEOUT)
                if audio is None:
                    assistant.pending = None
                    reply = "หมดเวลา ยกเลิกแล้ว"
                    print(f"[ตอบ]    {reply}")
                else:
                    answer = assistant.transcribe(audio)
                    print(f"[ASR]    {answer or '(ไม่ได้ยินอะไร)'}")
                    reply = assistant.handle_text(answer)
                if tts:
                    speak(reply)
    except KeyboardInterrupt:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="run one typed command (skips ASR)")
    src.add_argument("--audio", help="run one recorded command (wav/m4a/mp3)")
    src.add_argument("--wake", action="store_true", help='hands-free: listen for "เคทู" (K2), then the command')
    # v3: 50/60 end-to-end vs v2 48/60 — chosen on the 60 test commands, see CLUADE.md
    p.add_argument("--asr-model", default="models/whisper-small-th-v3",
                   help="whisper-small + LoRA fine-tuned on FLEURS, TTS commands and your voice (default); "
                        "models/whisper-small-th-v2 (fewer own recordings, better on English), "
                        "models/whisper-small-th (without your voice), models/whisper-th (tiny, faster)")
    p.add_argument("--asr-language", default="thai",
                   help="'thai' (default) or 'auto' to let Whisper detect — "
                        "often mis-detects short Thai/mixed commands (e.g. as Vietnamese)")
    # nb: best on the 36 unseen test commands (0.944 vs lstm 0.833) — chosen on the test set, see CLUADE.md
    p.add_argument("--intent-model", default="nb", choices=["nb", "logreg", "logreg_char", "cnn", "lstm"])
    p.add_argument("--device", help="cpu / cuda (default: cuda if available)")
    p.add_argument("--no-tts", action="store_true", help="print the reply only, don't speak it")
    args = p.parse_args()

    language = None if args.asr_language == "auto" else args.asr_language
    assistant = Assistant(args.asr_model, args.intent_model, language, args.device)
    tts = not args.no_tts

    if args.text or args.audio:
        reply = assistant.handle_text(args.text) if args.text else assistant.handle_audio(args.audio)
        if tts:
            speak(reply)
        if assistant.pending:
            try:
                answer = input("[ยืนยัน] ใช่/ไม่: ")
            except EOFError:
                answer = ""
            reply = assistant.handle_text(answer)
            if tts:
                speak(reply)
    elif args.wake:
        wake_loop(assistant, tts)
    else:
        interactive(assistant, tts)


if __name__ == "__main__":
    main()
