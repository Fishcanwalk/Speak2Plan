# Speak2Plan
Thai/English voice commands for Google Tasks &amp; Calendar, with a fine-tuned Whisper ASR and a self-trained CNN/LSTM intent classifier.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your Google OAuth client file at `credentials.json` (never commit it).

## Layout
- `data/` — datasets (MASSIVE, Common Voice/FLEURS, self-written sentences, self-recorded audio)
- `notebooks/` — `01_data_baseline`, `02_cnn_lstm`, `03_whisper_finetune` (run on Colab GPU)
- `src/` — `data`, `models`, `train`, `asr`, `google_api`, `pipeline`
- `models/` — trained checkpoints (not committed)

## Data credits
- MASSIVE (AmazonScience/massive), CC-BY-4.0
- Common Voice (Mozilla), CC0 / FLEURS (Google), CC-BY-4.0
