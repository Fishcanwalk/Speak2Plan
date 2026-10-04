"""Speak2Plan end-to-end: voice/text -> Whisper -> intent -> Google Tasks/Calendar -> spoken reply.

    python -m src.pipeline                                   # interactive: Enter = speak, or type a command
    python -m src.pipeline --wake                            # hands-free: say "เคทู ..." (K2) then the command
    python -m src.pipeline --text "เช็ค task ให้หน่อย"
    python -m src.pipeline --audio data/audio/cmd01.wav
    python -m src.pipeline --asr-model openai/whisper-small --intent-model logreg --no-tts
    python -m src.pipeline --wake --device cpu               # keep off the GPU (e.g. while training)

Read-only for now: check_tasks / check_calendar call Google; add_task / add_event / complete_task
are recognised but not executed (writing to Google needs a confirmation step — not built yet).
Mic uses `arecord`, playback uses `aplay` (alsa-utils). gTTS needs internet.
"""
import argparse
import io
import re
import subprocess
import tempfile
from collections import deque
from datetime import date, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from . import google_api
from .intent import IntentClassifier

SR = 16000
MIN_CONFIDENCE = 0.4   # below this the intent is treated as "didn't understand"
MAX_SPOKEN_ITEMS = 5   # read at most this many tasks/events aloud
CALENDAR_DAYS = 7

THAI_DAYS = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
THAI_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
               "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]

NOT_SUPPORTED = {
    "add_task": "เพิ่มงาน",
    "add_event": "เพิ่มนัด",
    "complete_task": "ทำเครื่องหมายว่างานเสร็จ",
}


# ---------- reply text ----------

def thai_day(d: date) -> str:
    delta = (d - date.today()).days
    if delta == 0:
        return "วันนี้"
    if delta == 1:
        return "พรุ่งนี้"
    return f"วัน{THAI_DAYS[d.weekday()]}ที่ {d.day} {THAI_MONTHS[d.month - 1]}"


def thai_when(start: str) -> str:
    """Calendar start: '2026-10-05T15:00:00+07:00' (timed) or '2026-10-05' (all-day)."""
    if "T" not in start:
        return f"{thai_day(date.fromisoformat(start))} ทั้งวัน"
    dt = datetime.fromisoformat(start).astimezone()
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


def reply_calendar() -> str:
    events = google_api.list_events(days=CALENDAR_DAYS)
    if not events:
        return f"ไม่มีนัดใน {CALENDAR_DAYS} วันข้างหน้า"
    items = [f"{thai_when(e['start'])} {e['summary']}" for e in events]
    return f"ใน {CALENDAR_DAYS} วันข้างหน้ามี {len(events)} นัด ได้แก่ {join_items(items)}"


def respond(intent: str, confidence: float) -> str:
    if confidence < MIN_CONFIDENCE:
        return "ไม่แน่ใจว่าหมายถึงอะไร ลองพูดใหม่อีกครั้งได้ไหม"
    try:
        if intent == "check_tasks":
            return reply_tasks()
        if intent == "check_calendar":
            return reply_calendar()
    except Exception as e:  # network / expired token — keep the demo alive
        print(f"  [Google error] {e}")
        return "เชื่อมต่อ Google ไม่สำเร็จ"
    if intent in NOT_SUPPORTED:
        return f"เข้าใจว่าต้องการ{NOT_SUPPORTED[intent]} แต่เวอร์ชันนี้ยังทำให้ไม่ได้"
    return "คำสั่งนี้ไม่เกี่ยวกับงานหรือปฏิทิน ลองถามเรื่องงานหรือนัดได้เลย"


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
# Whisper spells "เคทู" many ways (เกทู, เก-ทู, เคตุ, เคราะทู, เครีย์ธู, เคศู, เก่ต้, เกตทุ, K-2 ...),
# so match the sound pattern at the start of the utterance rather than one spelling.
WAKE_RE = re.compile(r"(?:hey|เฮ้?|เอ่อ)?(?:k|เค้?|เก่?|แค|เคราะ|เครีย์?)[ตด]?(?:2|two|tu|to|ทู|ตู|ทุ|ตุ|ต้|ตว|ธู|ศู|สู)")
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
    return rest if PARTICLES_RE.sub("", SKIP_RE.sub("", rest)) else ""


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
        probs = self.clf.predict_proba([text])[0]
        intent = max(probs, key=probs.get)
        print(f"[Intent] {intent} ({probs[intent]:.2f})")
        reply = respond(intent, probs[intent])
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
            line = input("\n[Enter] พูด / พิมพ์คำสั่ง / q ออก: ").strip()
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
    except KeyboardInterrupt:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--text", help="run one typed command (skips ASR)")
    src.add_argument("--audio", help="run one recorded command (wav/m4a/mp3)")
    src.add_argument("--wake", action="store_true", help='hands-free: listen for "เคทู" (K2), then the command')
    p.add_argument("--asr-model", default="models/whisper-th",
                   help="our fine-tuned tiny (default) or e.g. openai/whisper-small")
    p.add_argument("--asr-language", default="thai",
                   help="'thai' (default) or 'auto' to let Whisper detect — "
                        "often mis-detects short Thai/mixed commands (e.g. as Vietnamese)")
    p.add_argument("--intent-model", default="lstm", choices=["nb", "logreg", "logreg_char", "cnn", "lstm"])
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
    elif args.wake:
        wake_loop(assistant, tts)
    else:
        interactive(assistant, tts)


if __name__ == "__main__":
    main()
