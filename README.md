# Speak2Plan
Thai/English voice commands for Google Tasks &amp; Calendar, with a fine-tuned Whisper ASR and a self-trained CNN/LSTM intent classifier.

## Setup
```bash
uv venv --python 3.12 .venv        # or: python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt # or: pip install -r requirements.txt
```

Put your Google OAuth client file at `credentials.json` (never commit it).

## Part 1 — ASR: fine-tune Whisper (speech → text)
```bash
# download FLEURS Thai (~3 GB)
mkdir -p data/raw/fleurs_th
for s in train validation test; do
  curl -L -o data/raw/fleurs_th/$s.parquet \
    https://huggingface.co/api/datasets/google/fleurs/parquet/th_th/$s/0.parquet
done

python -m src.train_asr                                    # whisper-tiny -> models/whisper-th
python -m src.augment_asr                                  # gTTS + Whisper: noisy intent text, TTS clips
python -m src.train_asr --model openai/whisper-small --lora --tts-repeat 4 \
  --batch-size 4 --grad-accum 4 --gradient-checkpointing --max-steps 800 \
  --out models/whisper-small-th                            # best model, fits a 4 GB GPU
python -m src.asr eval --model openai/whisper-tiny --data fleurs   # before
python -m src.asr eval --model models/whisper-th   --data fleurs   # after
python -m src.asr eval --model models/whisper-th   --data data/audio/transcripts.csv  # own recordings (file,text)
```

## Part 2 — Intent classifier (text → intent), trained from scratch
```bash
python -m src.train_intent              # NB, LogReg, TextCNN, BiLSTM -> models/, reports/intent/
python -m src.intent --model lstm "เช็ค task ใน google ให้หน่อย"
```
MASSIVE is downloaded automatically. Add your own sentences to `data/custom_intents.csv`;
`data/generated_intents.csv` (Claude-written) and `data/augmented_intents.csv` (ASR-noisy, from
`src.augment_asr`) are added to training too.

## Test on your own voice
```bash
python -m src.record                 # test set:  data/audio/prompts.csv -> data/audio/*.wav + transcripts.csv
python -m src.record --set train     # train set: data/audio_train/ (different sentences) -> --own-repeat
python -m src.eval_e2e --asr-models models/whisper-th models/whisper-small-th
```

## Layout
- `data/` — `custom_intents.csv` (self-written), `generated_intents.csv` (Claude-written), `augmented_intents.csv`, `raw/` (downloaded datasets, not committed), `audio/` (own recordings)
- `src/` — `data`, `models`, `train_intent`, `intent`, `asr`, `train_asr`, `google_api`, `pipeline`
- `models/` — trained checkpoints (not committed)
- `reports/` — metrics, confusion matrices, CER results

## Data credits
- MASSIVE (Amazon, https://github.com/alexa/massive), CC-BY-4.0
- FLEURS (Google, `google/fleurs`), CC-BY-4.0
