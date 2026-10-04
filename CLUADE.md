# Speak2Plan — Project Context

## ภาพรวม
โปรเจกต์วิชา AI ทำระบบสั่งงานด้วยเสียงสำหรับ Google Tasks และ Google Calendar
โดย **train เอง 2 ส่วน**:
1. **ASR (เสียง → ข้อความ)** = fine-tune pre-trained Whisper ด้วย dataset เสียงไทยสาธารณะ
2. **Intent Classifier (ข้อความ → intent)** = train from scratch (CNN / LSTM)

ส่วนอื่นใช้ของสำเร็จรูป (rule-based วัน/เวลา, Google API, TTS)

**ชื่อโปรเจกต์:** Speak2Plan
**ชื่อเต็มในรายงาน (ตัวเลือกที่แนะนำ):**
- "Speak2Plan: Voice Command Intent Classification for Google Tasks and Calendar using CNN/LSTM"
- หรือแนวเปรียบเทียบ: "Comparative Study of Naive Bayes, TextCNN, and LSTM for Voice Command Intent Classification"

**เวลา:** ประมาณ 3 วัน (ตัดของเสริมออก เน้น pipeline ใช้งานได้จริง)
**ภาษา:** ไทย + อังกฤษ + ปนกัน (เช่น "เช็ค task ใน google ให้หน่อย")
**ตัวอย่างเป้าหมาย:** ผู้ใช้พูด "เปิดแล้วเช็ค task ใน google ให้หน่อย" → AI ไปดึง task แล้วตอบกลับเป็นเสียงว่ามี task อะไรบ้าง

## สถาปัตยกรรม
เสียง → **Whisper (fine-tune เอง)** → ข้อความ → **Intent Classifier (train เอง)** → เรียก Google Tasks/Calendar API → สร้างประโยคตอบ → TTS → เสียง

| ส่วน | วิธี | ต้องเทรน? |
|---|---|---|
| เสียง → ข้อความ (ASR) | Whisper fine-tune | **fine-tune pre-trained** |
| ข้อความ → Intent | TextCNN / LSTM (+ baseline NB, LogReg) | **train from scratch** |
| ดึงวัน/เวลา | rule-based (dateparser / PyThaiNLP) | ไม่ |
| Google Tasks/Calendar | API | ไม่ |
| ข้อความ → เสียง | gTTS | ไม่ |

**Fallback:** ถ้า fine-tune Whisper ไม่ทัน/ไม่ดีขึ้น เดโมใช้ Whisper สำเร็จรูปแทน (pipeline สลับโมเดลได้)

## Intent (6 ตัว)
`check_tasks`, `add_task`, `complete_task`, `check_calendar`, `add_event`, `other`

- เดโมเต็มๆ เฉพาะ read-only: `check_tasks`, `check_calendar` (ปลอดภัย)
- `add_event` / `add_task` ทำถ้ามีเวลา และต้องถามยืนยันก่อนเขียนข้อมูลเสมอ
- วัน/เวลาใน `add_event` ใช้ rule-based (`dateparser` / PyThaiNLP) ไม่ต้อง train เพิ่ม
- ระวังโมเดลสับสน add_task vs add_event (ใช้วิเคราะห์ใน confusion matrix ได้)

## Data — ASR (fine-tune Whisper)
- **หลัก:** FLEURS `th_th` (`google/fleurs`, CC-BY-4.0) — train 2,602 ประโยค (8.5 ชม.) / dev 439 / test 1,021
  โหลดเป็น parquet ลง `data/raw/fleurs_th/` (ดู `load_fleurs()` ใน `src/asr.py`)
  เป็นเสียงอ่านประโยคแนว Wikipedia ไม่ใช่คำสั่ง → domain ต่างจากงานจริง (เอาไปวิเคราะห์ในรายงาน)
- Common Voice ย้ายออกจาก Hugging Face แล้ว (repo เหลือแต่ README) — ไม่ได้ใช้
- **ชุดทดสอบ:** เสียงที่อัดเองเป็นคำสั่งจริง (ห้ามปนกับชุดเทรน)
- **วัดผล:** CER (หลัก เพราะภาษาไทยไม่มีช่องว่างระหว่างคำ) + WER หลังตัดคำ — เทียบ Whisper ก่อน/หลัง fine-tune
- รุ่น: `openai/whisper-tiny` (default, batch 8 × accum 2 พอดี RTX 2050 4GB) — `base` ได้ถ้าใช้ batch 4 × accum 4 + gradient checkpointing
- ต้องใช้ GPU — เครื่องนี้มี RTX 2050 4GB เทรน tiny ได้ (~0.77 step/s), batch 16 OOM เพราะ label ภาษาไทยยาว

## Data — Intent
- **หลัก:** MASSIVE (AmazonScience/massive, CC-BY-4.0) ภาษา `th-TH` และ `en-US`
  (~11.5k train / 2k dev / 3k test ต่อภาษา, 60 intent / 18 domain)
  HF repo ใช้ loading script ที่ `datasets` รุ่นใหม่ไม่รองรับ → `src/data.py` โหลด tar ต้นฉบับจาก S3 แทน
- แมป (ใน `src/data.py`): `lists_query`→check_tasks, `lists_createoradd`→add_task, `lists_remove`→complete_task,
  `calendar_query`→check_calendar, `calendar_set`→add_event, ที่เหลือ (รวม `calendar_remove`) → `other`
- `other` มี ~90% ของ data → สุ่มเก็บไว้ 12% (`--other-ratio`) กัน class imbalance
- MASSIVE ภาษาไทยเว้นวรรคระหว่างคำ แต่ข้อความจริง/Whisper ไม่เว้น → `normalize()` ลบช่องว่างระหว่างอักษรไทยก่อนตัดคำใหม่ด้วย PyThaiNLP
- **เขียนเองเพิ่ม** ~20–30 ประโยคต่อ intent ใน `data/custom_intents.csv` (text,intent) — ตอนนี้มีตัวอย่างตั้งต้น 5 ประโยค/intent ที่ Claude เขียน ผู้ใช้ต้องเขียนเพิ่ม
- **อัดเสียงตัวเอง** ~20–30 ประโยคเป็นชุดทดสอบ ส่งผ่าน Whisper เพื่อวัดความทนต่อ ASR error
- ในรายงานต้องแยกว่า data ส่วนไหน public / เขียนเอง และให้เครดิต MASSIVE

## แผนงาน 3 วัน
1. **วัน 1:** Data (MASSIVE + เขียนเอง) → ตัดคำ (PyThaiNLP) → TF-IDF → baseline: Naive Bayes + Logistic Regression
   *ทำ Google OAuth ควบคู่ตั้งแต่วันแรก (มักติดปัญหา)*
   *เริ่ม fine-tune Whisper บน Colab ทิ้งไว้ (รันนาน) + วัด CER ของ Whisper สำเร็จรูปเป็น baseline*
2. **วัน 2:** เทรน TextCNN และ/หรือ LSTM (PyTorch) → วัด accuracy, F1, confusion matrix เทียบ baseline
   → วัด CER Whisper ก่อน/หลัง fine-tune กับเสียงที่อัดเอง
   → ทดสอบ intent กับข้อความที่ Whisper (ทั้ง 2 รุ่น) ถอดจากเสียงจริง — ASR error กระทบ intent แค่ไหน
3. **วัน 3:** ต่อ pipeline Whisper → classifier → Google API → TTS, อัดวิดีโอ demo, เขียนรายงาน
   *(ถ้ากลัวไม่ทัน: เดโมแบบพิมพ์ข้อความก่อน แล้วค่อยใส่เสียง)*

**ถ้าเวลาตึง:** ทำ Intent ให้เสร็จก่อน — fine-tune Whisper เป็นส่วนรอง ใช้ Whisper สำเร็จรูปเป็น fallback

**ตัดออกเพื่อให้ทัน:** PCA/t-SNE, K-means, fine-tune WangchanBERTa, เทรน ASR จากศูนย์
(ถ้ามีเวลาเหลือค่อยทำ — ช่วยเพิ่มคะแนนส่วน Unsupervised/Dim. Reduction/NLP)

## ความเชื่อมโยงกับเนื้อหาวิชา (ใส่ในรายงาน)
| หัวข้อ | ใช้ตรงไหน |
|---|---|
| Intro | นิยามปัญหา supervised multi-class classification |
| Math for ML | embedding, softmax, cross-entropy, gradient descent |
| Data Processing | ทำความสะอาด, ตัดคำไทย, TF-IDF, tokenize/padding, split train/val/test |
| Regression | Logistic Regression เป็น baseline |
| Classification | จำแนก intent, accuracy/precision/recall/F1 |
| Bayesian inference | Naive Bayes baseline |
| Unsupervised (เสริม) | K-means หา intent ใหม่ |
| Dim. Reduction (เสริม) | PCA/t-SNE พล็อต embedding |
| NN & DL | layers, activation, backprop, dropout |
| CNNs | TextCNN จับวลีสำคัญ |
| RNNs | LSTM อ่านตามลำดับคำ |
| NLP | ตัดคำ, embedding, ASR error, ภาษาปน, fine-tune Whisper (transfer learning) |
| Eval & Opti | confusion matrix, tuning, early stopping, เทียบโมเดล, CER/WER ก่อน-หลัง fine-tune |
| RL | ไม่ได้ใช้ |

## ความรู้พื้นฐานที่คุยไปแล้ว
- **CNN (TextCNN):** filter เลื่อนจับวลี 2–3 คำ เร็ว เหมาะประโยคสั้น แต่ไม่เข้าใจบริบทไกล
- **LSTM:** อ่านทีละคำตามลำดับ มี gate/หน่วยความจำ เข้าใจบริบทและลำดับ แต่ช้ากว่า
- ถ้า data น้อย deep learning อาจแพ้ baseline TF-IDF — ไม่ใช่ปัญหา เอาไปวิเคราะห์ในรายงานได้
- **Train from scratch vs Fine-tune:** Intent = เริ่มน้ำหนักสุ่ม เรียนจาก MASSIVE ทั้งหมด;
  ASR = เริ่มจาก Whisper ที่เทรนมาแล้ว แค่ฝึกต่อด้วยเสียงไทย (ใช้ข้อมูล/เวลาน้อยกว่ามาก)
  — ในรายงานเรียกให้ถูก และแยกจาก hyperparameter tuning
- Whisper ไทยสำเร็จรูปแม่นพอสมควรอยู่แล้ว fine-tune อาจดีขึ้นไม่มาก — เอาไปวิเคราะห์ได้

## โครงสร้างโฟลเดอร์ที่วางไว้
```
Speak2Plan/
├── data/
├── models/            # checkpoint ที่เทรนแล้ว (ไม่ commit)
├── notebooks/         # (ทางเลือก ไว้ทำกราฟ/วิเคราะห์ลงรายงาน — โค้ดเทรนหลักอยู่ใน src/)
├── reports/           # ผลการเทรน/ประเมิน (metrics, confusion matrix, CER) — commit ได้
├── src/
│   ├── data.py          # โหลด MASSIVE + custom, แมป intent, normalize/tokenize
│   ├── models.py        # TextCNN, BiLSTM, Vocab
│   ├── train_intent.py  # ส่วนที่ 2: เทรน NB / LogReg / CNN / LSTM + ประเมิน
│   ├── intent.py        # โหลดโมเดล intent แล้วทำนาย
│   ├── asr.py           # โหลด FLEURS, Transcriber, CER/WER, CLI eval/transcribe
│   ├── train_asr.py     # ส่วนที่ 1: fine-tune Whisper
│   ├── google_api.py, pipeline.py  # (ยังว่าง)
├── credentials.json   # ห้าม commit
├── .gitignore  (.venv/, credentials.json, token.json, __pycache__/, models/)
├── requirements.txt
└── README.md
```
Dependencies หลัก: torch, datasets, scikit-learn, pandas, matplotlib, seaborn, pythainlp,
transformers, accelerate, evaluate, jiwer, librosa/soundfile (fine-tune Whisper ผ่าน Hugging Face),
google-api-python-client, google-auth-oauthlib, gTTS, jupyter
(CNN/LSTM ขนาดเล็กเทรนบน CPU ได้; fine-tune Whisper ต้องใช้ GPU/Colab)

## สถานะปัจจุบัน
- โค้ดเทรนทั้ง 2 ส่วนเสร็จและรันได้แล้ว (venv: `uv venv --python 3.12`, torch 2.14 + CUDA, transformers 5.x)
- Intent (test, MASSIVE th+en): NB acc 0.838 / F1 0.815, **LogReg acc 0.860 / F1 0.844**, CNN 0.811 / 0.808, LSTM 0.828 / 0.815
  → baseline ชนะ deep learning (data น้อย) ; check_calendar อ่อนสุด (F1 ~0.70–0.76)
- ASR (FLEURS test 1,021 ประโยค): whisper-tiny ก่อน fine-tune **CER 0.638 / WER 1.417** → หลัง fine-tune (`models/whisper-th`) **CER 0.231 / WER 0.659**
  (1,500 steps, ~9 epoch, best dev CER 0.211) — WER > 1 ก่อน fine-tune เพราะ tiny หลอน/พูดวนซ้ำ (ข้อความยาวเกิน 2 เท่า 52 ประโยค → เหลือ 10)
  ผลอยู่ใน `reports/asr/` ; ยังผิดคำศัพท์เฉพาะ/คำทับศัพท์บ่อย
- ระวัง: เครื่อง suspend ระหว่างเทรน → GPU ค้าง ; รันด้วย `systemd-inhibit --what=sleep:idle ...` และใช้ `--resume` ต่อจาก checkpoint ได้
- ยังไม่ได้: Google OAuth, pipeline, TTS, อัดเสียงตัวเอง, เขียน custom data เพิ่ม

## สิ่งที่อยากให้ Claude ช่วยต่อ (ลำดับแนะนำ)
1. สร้างโครงโปรเจกต์ + environment + requirements
2. ทำ Google OAuth และทดสอบ `list tasks` / `list events`
3. โหลด MASSIVE → แมป 6 intent → baseline (Naive Bayes, LogReg)
4. Notebook fine-tune Whisper (Colab) + วัด CER ก่อน/หลัง
5. เทรน TextCNN / LSTM → ประเมินผล
6. ต่อ Whisper (fine-tuned) + pipeline + TTS, เตรียมรายงาน

## ข้อตกลงการทำงาน
- ผู้ใช้พิมพ์/คุยเป็นภาษาไทย ตอบเป็นภาษาไทย ศัพท์เทคนิคใช้อังกฤษได้
- อธิบายให้เข้าใจได้ง่าย เพราะเป็นโปรเจกต์เรียน และต้องเขียนรายงานเอง
- อย่า commit `credentials.json` / `token.json`
- การเขียน/แก้ข้อมูลใน Google ต้องถามยืนยันก่อน