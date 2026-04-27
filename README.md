# Early Mental Health Signal Detector

**Privacy-first NLP tool for school counselors and clinicians.**

> ⚠️ **Professional use only.** This is a clinical decision-support tool, not a diagnostic instrument and not for self-assessment. All analysis runs locally — no text is ever stored or transmitted.

---

## Overview

| Component              | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| `models/model.py`      | BERT + LIWC linguistic features, 4-class classifier   |
| `models/train.py`      | Full training pipeline with evaluation plots          |
| `pipeline/features.py` | LIWC-style feature extraction (local wordlists)       |
| `pipeline/analyze.py`  | Inference + LIME phrase highlighting                  |
| `app.py`               | Streamlit dashboard (counselor view + trend analysis) |

**Labels:** `depression` · `anxiety` · `crisis` · `neutral`

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get training data

#### Option A — Reddit Mental Health Dataset (HuggingFace, auto-downloaded)

The training script automatically tries to fetch a Reddit mental health dataset from HuggingFace. No extra steps needed if you have internet access during training.

#### Option B — CLPsych 2015 Dataset (recommended for best quality)

CLPsych 2015 requires free registration at [clpsych.org](https://clpsych.org).
After downloading, prepare a CSV with columns `text` and `label`:

```
text,label
"I've been feeling hopeless all week...",depression
"My anxiety has been overwhelming...",anxiety
"I don't want to be here anymore...",crisis
"Had a good day today...",neutral
```

Pass it to the training script with `--data-dir`.

#### Option C — Synthetic demo (smoke-test only)

If no dataset is found, a tiny synthetic dataset is used automatically. **Do not deploy a model trained on synthetic data.**

### 3. Train the model

```bash
# Auto-download from HuggingFace
python -m models.train --output-dir models/saved --epochs 5

# Local CSV (CLPsych or custom)
python -m models.train --data-dir data/clpsych.csv --output-dir models/saved --epochs 5
```

Training takes ~30 min on a GPU, ~4 hours on CPU for a real dataset.
Saved artefacts: `models/saved/bert/`, `models/saved/tokenizer/`, `models/saved/head.pt`, evaluation plots.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Model Architecture

```
Input text
   │
   ├─► BertTokenizer → BERT-base-uncased → [CLS] embedding (768-d)
   │
   └─► LIWC-style features (8-d) → Linear(8→32) → ReLU
                                                          │
                                   Concat(768+32) = 800-d │
                                                          ▼
                                        Linear(800→256) → GELU → Dropout(0.3)
                                        Linear(256→4)   → logits
                                        Softmax         → probabilities
```

**LIWC-style features (8):**

1. Word count (normalised)
2. First-person pronoun rate
3. Negative-word ratio
4. Positive-word ratio
5. Crisis-vocabulary ratio
6. Cognitive-processing word rate
7. Net negative sentiment (neg − pos, normalised)
8. Average sentence length (normalised)

**Explainability:** LIME (Local Interpretable Model-agnostic Explanations) highlights which words contributed most to the predicted risk category. Yellow = risk signal; green = protective/resilience signal.

---

## Training Configuration

| Hyperparameter      | Value               |
| ------------------- | ------------------- |
| Base model          | `bert-base-uncased` |
| Optimizer           | AdamW               |
| Learning rate       | 2e-5                |
| Epochs              | 5                   |
| Batch size          | 16                  |
| Max sequence length | 256                 |
| Warmup steps        | 10% of total        |
| Grad clip           | 1.0                 |
| Split               | 80 / 10 / 10        |

**Evaluation metrics:** F1 per class, macro-F1, ROC-AUC (OvR), precision-recall curves.
Plots are saved to `models/saved/` after training.

---

## Deployment in a School or Clinic Setting

1. **Air-gapped deployment:** Copy the full project folder (including `models/saved/`) to the counselor's local machine. No internet connection is required at inference time.
2. **Run locally:** `streamlit run app.py` — the dashboard binds to `localhost` only by default.
3. **No database:** There is no database, no logging backend, and no network call in the inference path.
4. **Access control:** Restrict filesystem access to `models/saved/` to authorized personnel. The trained model itself does not contain any training-set text.
5. **Session isolation:** Each browser tab is a separate Streamlit session. Closing the tab clears all session state.
6. **Audit trail:** If your institution requires an audit trail of _who_ ran an analysis (not _what_ text was analyzed), implement OS-level access logging on the machine — do not log text content.

---

## Ethics & Responsible Use

### Intended Use

This tool is designed to assist trained mental health professionals in identifying linguistic patterns that may warrant further assessment. It is a **second opinion**, not a replacement for clinical judgment.

### What this tool does NOT do

- Diagnose any mental health condition
- Replace direct clinical assessment or conversation
- Provide medical advice
- Guarantee detection of all at-risk individuals (false negatives are possible and expected)

### Known Limitations and Biases

| Limitation                           | Detail                                                                                                                                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Training data bias**               | Reddit-sourced data skews younger, English-speaking, and Western. The model may underperform for older adults, non-native English writers, or culturally specific expressions of distress. |
| **Text-only signal**                 | Vocal tone, body language, and contextual history — often the most important clinical cues — are invisible to this model.                                                                  |
| **Label ambiguity**                  | Depression and anxiety frequently co-occur; the model's categorical labels are a simplification of a continuum.                                                                            |
| **Crisis detection recall priority** | The model is tuned toward recall for the `crisis` class (fewer false negatives) at the cost of precision. Expect some false-positive crisis flags.                                         |
| **CLPsych 2015 domain shift**        | CLPsych 2015 was drawn from a 2013 Twitter/Reddit snapshot; language norms shift over time.                                                                                                |
| **Adversarial robustness**           | The model is not hardened against deliberate obfuscation.                                                                                                                                  |

### Data Handling Requirements for Deployers

- Input text must be de-identified before analysis (remove names, dates, locations, relationship identifiers).
- Do not store, log, or transmit analysis outputs alongside re-identifiable metadata.
- Follow your institution's IRB/ethics board guidelines and applicable laws (FERPA, HIPAA, GDPR as relevant).
- Obtain appropriate consent where required by institutional policy.

### Mandatory Human Oversight

**Any HIGH or MODERATE risk flag must be reviewed by a qualified clinician before any intervention is initiated.** The tool output is advisory only.

---

## Model Card

| Field           | Value                                                                       |
| --------------- | --------------------------------------------------------------------------- |
| Model type      | BERT-base-uncased + MLP head                                                |
| Task            | 4-class text classification                                                 |
| Languages       | English                                                                     |
| Training data   | Reddit Mental Health subreddits + (optional) CLPsych 2015                   |
| Evaluation data | 10% held-out split from same distribution                                   |
| License         | For research/clinical use only — not for commercial redistribution          |
| Contact         | Deploy under institutional oversight; do not release model weights publicly |

---

## File Structure

```
Health/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
├── models/
│   ├── model.py            # BertWithLinguisticFeatures
│   ├── train.py            # Training script
│   └── saved/              # Trained model (generated by training)
│       ├── bert/           # BERT weights (save_pretrained format)
│       ├── tokenizer/      # BertTokenizer
│       ├── head.pt         # Classifier + projection weights
│       ├── test_metrics.json
│       ├── f1_per_class.png
│       └── precision_recall.png
└── pipeline/
    ├── features.py         # LIWC-style feature extraction
    └── analyze.py          # Inference + LIME explainability
```
