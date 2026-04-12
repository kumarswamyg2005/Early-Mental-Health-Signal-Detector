"""
Training script: fine-tune BERT + LIWC features for mental health signal detection.

Datasets (tried in order):
  1. imancheema/Reddit-Mental-Health-Dataset  (HuggingFace)
  2. ShreyaR/reddit-mental-health-dataset     (HuggingFace)
  3. CLPsych 2015 via local CSV  (--data-dir path/to/clpsych.csv)
  4. Synthetic demo data         (fallback for smoke-testing)

Usage:
  python -m models.train --output-dir models/saved --epochs 5
  python -m models.train --data-dir data/clpsych.csv --output-dir models/saved
"""

import os
import json
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.model import MentalHealthClassifier, LABEL_NAMES, NUM_LING_FEATURES
from pipeline.features import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LABEL2IDX = {name: i for i, name in enumerate(LABEL_NAMES)}

# Subreddit → label mappings for Reddit datasets
_SUBREDDIT_MAP: dict[str, str] = {
    # depression
    "depression": "depression", "depressed": "depression",
    "mentalhealth": "depression", "mentalillness": "depression",
    "bipolar": "depression", "ptsd": "depression",
    # anxiety
    "anxiety": "anxiety", "panicattack": "anxiety",
    "panicattacks": "anxiety", "socialanxiety": "anxiety",
    "ocd": "anxiety", "phobia": "anxiety",
    # crisis
    "suicidewatch": "crisis", "suicide": "crisis",
    "selfharm": "crisis", "crisisresources": "crisis",
    # neutral
    "casualconversation": "neutral", "askreddit": "neutral",
    "iama": "neutral", "worldnews": "neutral",
    "jokes": "neutral", "todayilearned": "neutral",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _map_reddit_label(subreddit: str) -> str | None:
    return _SUBREDDIT_MAP.get(subreddit.lower().replace(" ", ""))


def _load_hf_dataset(name: str) -> pd.DataFrame | None:
    try:
        from datasets import load_dataset
        log.info(f"Trying HuggingFace dataset: {name}")
        ds = load_dataset(name)
        split = list(ds.keys())[0]
        df = ds[split].to_pandas()

        # Detect text column
        text_col = next((c for c in df.columns if "text" in c.lower() or "body" in c.lower()), None)
        label_col = next(
            (c for c in df.columns if "label" in c.lower() or "subreddit" in c.lower() or "class" in c.lower()),
            None,
        )
        if text_col is None or label_col is None:
            log.warning(f"Could not detect text/label columns in {name}: {list(df.columns)}")
            return None

        df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "raw_label"})
        df["label"] = df["raw_label"].apply(
            lambda x: _map_reddit_label(str(x)) if _map_reddit_label(str(x)) else None
        )
        df = df.dropna(subset=["label"])
        df["label_idx"] = df["label"].map(LABEL2IDX)
        log.info(f"Loaded {len(df)} rows from {name}. Distribution:\n{df['label'].value_counts()}")
        return df[["text", "label", "label_idx"]]
    except Exception as e:
        log.warning(f"Failed to load {name}: {e}")
        return None


def _load_csv(path: str) -> pd.DataFrame:
    """Load a local CSV. Expected columns: text, label (depression/anxiety/crisis/neutral)."""
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns {required}, got {list(df.columns)}")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(LABEL_NAMES)]
    df["label_idx"] = df["label"].map(LABEL2IDX)
    log.info(f"Loaded {len(df)} rows from {path}. Distribution:\n{df['label'].value_counts()}")
    return df[["text", "label", "label_idx"]]


def _synthetic_dataset(n: int = 2000) -> pd.DataFrame:
    """Tiny synthetic dataset for smoke-testing the pipeline."""
    import random
    random.seed(42)
    templates = {
        "depression": [
            "I feel so hopeless and empty. Nothing brings me joy anymore.",
            "I've been sad for weeks. I don't see the point in anything.",
            "Everything feels heavy. I can't get out of bed.",
        ],
        "anxiety": [
            "I can't stop worrying about everything. My heart races constantly.",
            "The panic attacks are getting worse. I'm afraid to leave the house.",
            "My mind won't stop. I'm overwhelmed and can't focus.",
        ],
        "crisis": [
            "I've been thinking about ending my life. I have a plan.",
            "I don't want to be here anymore. I'm going to hurt myself tonight.",
            "Nobody would miss me if I disappeared. I'm going to do it.",
        ],
        "neutral": [
            "Had a great day today. Went for a walk and felt refreshed.",
            "Work was busy but I managed to get everything done.",
            "Tried a new recipe today. It turned out really well!",
        ],
    }
    rows = []
    per_class = n // 4
    for label, texts in templates.items():
        for _ in range(per_class):
            rows.append({"text": random.choice(texts), "label": label, "label_idx": LABEL2IDX[label]})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    log.warning("Using SYNTHETIC demo data. Do NOT use this model in production.")
    return df


def load_data(data_dir: str | None) -> pd.DataFrame:
    if data_dir:
        return _load_csv(data_dir)
    for name in [
        "imancheema/Reddit-Mental-Health-Dataset",
        "ShreyaR/reddit-mental-health-dataset",
    ]:
        df = _load_hf_dataset(name)
        if df is not None and len(df) >= 500:
            return df
    return _synthetic_dataset()


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MentalHealthDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, max_length: int = 512):
        self.texts = df["text"].tolist()
        self.labels = df["label_idx"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ling = torch.tensor(extract_features(self.texts[idx]), dtype=torch.float32)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ling_features": ling,
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["ling_features"].to(device),
            )
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    per_class_f1 = f1_score(all_labels, all_preds, average=None, labels=list(range(4)), zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    y_bin = label_binarize(all_labels, classes=list(range(4)))
    try:
        roc_auc = roc_auc_score(y_bin, all_probs, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    return {
        "macro_f1": macro_f1,
        "per_class_f1": {name: round(float(per_class_f1[i]), 4) for i, name in enumerate(LABEL_NAMES)},
        "roc_auc": round(roc_auc, 4),
        "report": classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0),
        "probs": all_probs,
        "labels": all_labels,
    }


def save_plots(metrics: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Per-class F1 bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    f1s = metrics["per_class_f1"]
    ax.bar(f1s.keys(), f1s.values(), color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "f1_per_class.png"), dpi=120)
    plt.close(fig)

    # Precision-Recall curves (one-vs-rest)
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(metrics["labels"], classes=list(range(4)))
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], metrics["probs"][:, i])
        ap = average_precision_score(y_bin[:, i], metrics["probs"][:, i])
        ax.plot(rec, prec, color=color, label=f"{name} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "precision_recall.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = MentalHealthClassifier(num_ling_features=NUM_LING_FEATURES).to(device)

    df = load_data(args.data_dir)

    # 80 / 10 / 10 split
    train_df, tmp = train_test_split(df, test_size=0.2, stratify=df["label_idx"], random_state=42)
    val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp["label_idx"], random_state=42)
    log.info(f"Split — train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

    train_ds = MentalHealthDataset(train_df, tokenizer, max_length=args.max_length)
    val_ds   = MentalHealthDataset(val_df,   tokenizer, max_length=args.max_length)
    test_ds  = MentalHealthDataset(test_df,  tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,            num_workers=args.workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2,            num_workers=args.workers)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["ling_features"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_m = evaluate(model, val_loader, device)
        log.info(
            f"Epoch {epoch+1} — loss: {avg_loss:.4f} | val macro-F1: {val_m['macro_f1']:.4f} "
            f"| ROC-AUC: {val_m['roc_auc']:.4f}"
        )

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            model.save(args.output_dir)
            tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
            log.info(f"  ✓ Saved best model (val F1={best_val_f1:.4f})")

    # Final test evaluation
    log.info("\n--- Test Set Evaluation ---")
    best_model = MentalHealthClassifier.load(args.output_dir).to(device)
    test_m = evaluate(best_model, test_loader, device)
    log.info(f"Test macro-F1: {test_m['macro_f1']:.4f} | ROC-AUC: {test_m['roc_auc']:.4f}")
    log.info(f"\n{test_m['report']}")

    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in test_m.items() if k not in ("probs", "labels")}, f, indent=2)
    save_plots(test_m, args.output_dir)
    log.info(f"Metrics and plots saved to {args.output_dir}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mental health signal detector")
    parser.add_argument("--data-dir",    default=None,           help="Path to local CSV (text, label columns)")
    parser.add_argument("--output-dir",  default="models/saved", help="Where to save the trained model")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--max-length",  type=int,   default=256)
    parser.add_argument("--workers",     type=int,   default=2)
    args = parser.parse_args()
    train(args)
