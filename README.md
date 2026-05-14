# MindSense — Counselor Support Tool

**Privacy-first wellbeing assessment for licensed mental health professionals and school counselors.**

> ⚠️ **Professional use only.** This is a clinical decision-support tool — not a diagnostic instrument and not for self-assessment. All inference runs locally. No text is ever stored, logged, or transmitted.

---

## Overview

MindSense helps clinicians identify early linguistic patterns associated with mental health signals in de-identified text. It is built on a local NLP pipeline (no cloud API calls) with a clean React interface and a FastAPI backend.

| Component | Description |
| --- | --- |
| `models/model.py` | BERT + LIWC-style linguistic features, 4-class classifier |
| `models/train.py` | Training pipeline with evaluation plots |
| `pipeline/features.py` | Linguistic feature extraction (local word lists) |
| `pipeline/analyze.py` | Inference + importance-weighted phrase highlights |
| `backend/main.py` | FastAPI REST API (wraps the inference pipeline) |
| `frontend/` | React + Vite counselor interface |

**Assessment categories:** `depression` · `anxiety` · `crisis` · `neutral`

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Get training data

#### Option A — HuggingFace (auto-downloaded)

The training script fetches a Reddit mental health dataset automatically. No extra steps needed if you have internet access during training.

#### Option B — CLPsych 2015 (recommended for best quality)

CLPsych 2015 requires free registration at [clpsych.org](https://clpsych.org). After downloading, prepare a CSV with columns `text` and `label`:

```
text,label
"I've been feeling hopeless all week...",depression
"My anxiety has been overwhelming...",anxiety
"I don't want to be here anymore...",crisis
"Had a good day today...",neutral
```

Pass the path to the training script with `--data-dir`.

#### Option C — Synthetic demo (smoke-test only)

If no dataset is found, a tiny synthetic dataset is used. **Do not deploy a model trained on synthetic data.**

### 3. Train the model

```bash
# Auto-download from HuggingFace
python -m models.train --output-dir models/saved --epochs 5

# Local CSV (CLPsych or custom)
python -m models.train --data-dir data/clpsych.csv --output-dir models/saved --epochs 5
```

Training takes ~30 min on a GPU, ~4 hours on CPU for a real dataset.
Saved artefacts: `models/saved/bert/`, `models/saved/tokenizer/`, `models/saved/head.pt`, evaluation plots.

### 4. Start the backend

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`. The `/health` endpoint confirms the model is loaded.

### 5. Start the frontend

```bash
cd frontend
npm install        # first time only
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

> Both the backend and frontend must be running simultaneously. The frontend connects to the backend at `http://127.0.0.1:8000`.

---

## Running Both Together (convenience script)

```bash
chmod +x start.sh
./start.sh
```

---

## API Reference

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness check; reports whether model is loaded |
| POST | `/analyze` | Assess a single text entry |
| POST | `/analyze/batch` | Assess up to 50 entries for trend view |

**`POST /analyze` body:**

```json
{
  "text": "de-identified text here",
  "num_lime_samples": 50
}
```

**`POST /analyze/batch` body:**

```json
{
  "texts": ["entry one", "entry two", "entry three"]
}
```

---

## Model Architecture

```
Input text
   │
   ├─► BertTokenizer → BERT-base-uncased → [CLS] embedding (768-d)
   │
   └─► Linguistic features (8-d) → Linear(8→32) → ReLU
                                                         │
                                  Concat(768+32) = 800-d │
                                                         ▼
                                       Linear(800→256) → GELU → Dropout(0.3)
                                       Linear(256→4)   → logits
                                       Softmax         → probabilities
```

**Linguistic features (8):**

1. Word count (normalised)
2. First-person pronoun rate
3. Negative-word ratio
4. Positive-word ratio
5. Crisis-vocabulary ratio
6. Cognitive-processing word rate
7. Net negative sentiment (neg − pos, normalised)
8. Average sentence length (normalised)

**Phrase highlighting:** Word-importance weights show which tokens most influenced the prediction. Amber highlight = risk-associated language; green highlight = protective / resilience language.

---

## Training Configuration

| Hyperparameter | Value |
| --- | --- |
| Base model | `bert-base-uncased` |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Epochs | 5 |
| Batch size | 16 |
| Max sequence length | 256 |
| Warmup steps | 10% of total |
| Gradient clipping | 1.0 |
| Train / val / test | 80 / 10 / 10 |

**Evaluation metrics:** F1 per class, macro-F1, ROC-AUC (OvR), precision-recall curves.
Plots are saved to `models/saved/` after training.

---

## Deployment

### Air-gapped (recommended for clinical settings)

1. Copy the full project directory (including `models/saved/`) to the target machine.
2. Install Python dependencies: `pip install -r requirements.txt`
3. Install frontend dependencies: `cd frontend && npm install`
4. Start the backend (`uvicorn`) and frontend (`npm run dev`) — no internet required at runtime.
5. The app binds to `localhost` only by default; do not expose it externally.

### Access control

- Restrict filesystem access to `models/saved/` to authorised personnel.
- The trained model does not contain any training-set text.
- If your institution requires an audit trail of *who* accessed the tool (not *what* text was analysed), implement OS-level access logging — do not log text content.

### Session isolation

Each browser tab is an independent session. Closing the tab clears all client-side state. The backend holds no per-session state.

---

## Ethics & Responsible Use

### Intended use

This tool assists trained mental health professionals in identifying linguistic patterns that may warrant further clinical attention. It is a **second opinion**, not a replacement for direct assessment and professional judgment.

### What this tool does NOT do

- Diagnose any mental health condition
- Replace clinical assessment or therapeutic conversation
- Provide medical advice of any kind
- Guarantee detection of all at-risk individuals — false negatives are possible and expected

### Known limitations and biases

| Limitation | Detail |
| --- | --- |
| **Training data bias** | Reddit-sourced data skews younger, English-speaking, and Western. The model may underperform for older adults, non-native English writers, or culturally specific expressions of distress. |
| **Text-only signal** | Vocal tone, body language, and contextual history — often the most clinically important cues — are invisible to this model. |
| **Label ambiguity** | Depression and anxiety frequently co-occur; the model's four-category output is a simplification of a clinical continuum. |
| **Crisis recall priority** | The model is tuned toward recall for the `crisis` class (fewer false negatives) at the cost of precision. Expect some false-positive crisis flags. |
| **Domain shift** | Training data reflects language norms from a specific time period; language evolves. |
| **Adversarial robustness** | The model is not hardened against deliberate obfuscation. |

### Data handling requirements

- De-identify all text before analysis (remove names, dates, locations, relationship identifiers).
- Do not store, log, or transmit assessment outputs alongside re-identifiable metadata.
- Follow applicable regulations (FERPA, HIPAA, GDPR, DPDP Act, or equivalent) and your institution's IRB/ethics policy.
- Obtain appropriate consent where required by institutional or local policy.

### Mandatory human oversight

**Any HIGH or MODERATE risk flag must be reviewed by a qualified clinician before any intervention is initiated.** The tool output is advisory only.

---

## Model Card

| Field | Value |
| --- | --- |
| Model type | BERT-base-uncased + MLP head with linguistic features |
| Task | 4-class text classification |
| Languages | English |
| Training data | Reddit Mental Health subreddits + (optional) CLPsych 2015 |
| Evaluation data | 10% held-out split from same distribution |
| License | Research / clinical use only — not for commercial redistribution |
| Contact | Deploy under institutional oversight; do not release model weights publicly |

---

## File Structure

```
Health/
├── backend/
│   └── main.py             # FastAPI REST API
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css       # Design system (Quiet Sanctuary palette)
│   │   └── components/
│   │       ├── Header.jsx / .css
│   │       ├── Sidebar.jsx / .css
│   │       ├── AnalyzeView.jsx / .css
│   │       ├── TrendView.jsx / .css
│   │       ├── AboutView.jsx / .css
│   │       ├── ResultPanel.jsx / .css
│   │       ├── RiskBadge.jsx / .css
│   │       ├── ScoreBar.jsx / .css
│   │       ├── WordHighlight.jsx / .css
│   │       └── ResourceCards.jsx / .css
│   ├── package.json
│   └── vite.config.js
├── models/
│   ├── model.py            # BertWithLinguisticFeatures
│   ├── train.py            # Training script
│   └── saved/              # Trained model (generated by training)
│       ├── bert/
│       ├── tokenizer/
│       ├── head.pt
│       ├── test_metrics.json
│       ├── f1_per_class.png
│       └── precision_recall.png
├── pipeline/
│   ├── features.py         # Linguistic feature extraction
│   └── analyze.py          # Inference + phrase importance weights
├── requirements.txt
└── start.sh                # Convenience launcher (backend + frontend)
```
