"""
Inference pipeline: text → risk scores + LIME phrase highlights + linguistic features.
All processing is local. No text is stored or transmitted.
"""

import os
import re
import logging
from typing import Optional
import numpy as np
import torch
from transformers import BertTokenizer

from models.model import MentalHealthClassifier, LABEL_NAMES, NUM_LING_FEATURES
from pipeline.features import extract_features, describe_features, top_notable_features

log = logging.getLogger(__name__)

_RISK_ORDER = ["crisis", "depression", "anxiety", "neutral"]

# ---------------------------------------------------------------------------
# Risk-level thresholds
# ---------------------------------------------------------------------------

def get_risk_level(scores: dict[str, float]) -> tuple[str, str]:
    """Return (level, primary_label) where level ∈ HIGH/MODERATE/LOW/MINIMAL."""
    crisis = scores.get("crisis", 0)
    dep    = scores.get("depression", 0)
    anx    = scores.get("anxiety", 0)

    if crisis >= 0.45:
        return "HIGH", "crisis"
    if crisis >= 0.25 or dep >= 0.50 or anx >= 0.50:
        primary = "depression" if dep >= anx else "anxiety"
        return "MODERATE", primary
    if dep >= 0.25 or anx >= 0.25:
        primary = "depression" if dep >= anx else "anxiety"
        return "LOW", primary
    return "MINIMAL", "neutral"


RISK_RESOURCES: dict[str, list[dict]] = {
    "HIGH": [
        {"name": "iCall (TISS)",               "detail": "9152987821 — Mon–Sat, 8 am–10 pm"},
        {"name": "Vandrevala Foundation",      "detail": "1860-2662-345 — 24/7, free & confidential"},
        {"name": "Snehi",                      "detail": "044-24640050 — 24/7 emotional support"},
        {"name": "National Emergency (112)",   "detail": "Call 112 for police/ambulance"},
        {"name": "NIMHANS Helpline",           "detail": "080-46110007"},
    ],
    "MODERATE": [
        {"name": "iCall (TISS)",               "detail": "9152987821 — counselling & referral"},
        {"name": "Vandrevala Foundation",      "detail": "1860-2662-345 — 24/7"},
        {"name": "Fortis Stress Helpline",     "detail": "8376804102"},
        {"name": "iCall Online Therapy",       "detail": "icallhelpline.org — low-cost therapy"},
    ],
    "LOW": [
        {"name": "School/Clinic Counselor",    "detail": "Schedule an in-person session"},
        {"name": "iCall",                      "detail": "9152987821 — Mon–Sat, 8 am–10 pm"},
        {"name": "YourDOST",                   "detail": "yourdost.com — anonymous online counselling"},
    ],
    "MINIMAL": [
        {"name": "Wysa / InnerHour",           "detail": "AI-assisted mental wellness apps (India-based)"},
        {"name": "NIMHANS Mental Health",      "detail": "nimhans.ac.in — resources & self-help guides"},
    ],
}


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class MentalHealthAnalyzer:
    """Load a trained model and run inference with LIME-based phrase highlighting."""

    def __init__(self, model_path: str = "models/saved"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MentalHealthClassifier.load(model_path).to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
        self._lime_explainer = None  # lazy init

    # ------------------------------------------------------------------
    # Internal prediction (accepts list of raw strings — required by LIME)
    # ------------------------------------------------------------------

    def _predict_batch(self, texts: list[str]) -> np.ndarray:
        """Return [N, 4] probability array. Used by LIME and direct inference."""
        all_probs = []
        for text in texts:
            ling = torch.tensor([extract_features(text)], dtype=torch.float32, device=self.device)
            enc = self.tokenizer(
                text,
                max_length=256,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids, attention_mask, ling)
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            all_probs.append(probs)
        return np.array(all_probs)

    # ------------------------------------------------------------------
    # LIME word-importance
    # ------------------------------------------------------------------

    def _get_lime_weights(
        self,
        text: str,
        label_idx: int,
        num_samples: int = 50,
    ) -> dict[str, float]:
        """Return {word: weight} for the given label using LIME."""
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError:
            log.warning("lime not installed. Run: pip install lime")
            return {}

        if self._lime_explainer is None:
            self._lime_explainer = LimeTextExplainer(
                class_names=LABEL_NAMES, random_state=42
            )

        try:
            exp = self._lime_explainer.explain_instance(
                text,
                self._predict_batch,
                labels=[label_idx],
                num_features=25,
                num_samples=num_samples,
            )
            return {word.lower(): weight for word, weight in exp.as_list(label=label_idx)}
        except Exception as e:
            log.warning(f"LIME explanation failed: {e}")
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str, num_lime_samples: int = 50) -> dict:
        """
        Returns:
          scores          — {label: probability}
          risk_level      — HIGH / MODERATE / LOW / MINIMAL
          primary_label   — most prominent risk label
          word_weights    — {word: lime_weight}  (positive = risk-contributing)
          top_features    — list of 3 notable linguistic features
          feature_values  — raw feature dict
          resources       — recommended resources for this risk level
        """
        probs  = self._predict_batch([text])[0]
        scores = {name: round(float(p), 4) for name, p in zip(LABEL_NAMES, probs)}

        risk_level, primary_label = get_risk_level(scores)

        # LIME on the highest-risk non-neutral label
        risk_label = next(
            (l for l in _RISK_ORDER if scores[l] > 0.15 and l != "neutral"),
            LABEL_NAMES[int(np.argmax(probs))],
        )
        label_idx = LABEL_NAMES.index(risk_label)
        word_weights = self._get_lime_weights(text, label_idx, num_samples=num_lime_samples)

        ling = extract_features(text)

        return {
            "scores":        scores,
            "risk_level":    risk_level,
            "primary_label": primary_label,
            "word_weights":  word_weights,
            "top_features":  top_notable_features(ling),
            "feature_values": describe_features(ling),
            "resources":     RISK_RESOURCES[risk_level],
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Lightweight batch inference (no LIME, no feature explanations) for trend view."""
        all_probs = self._predict_batch(texts)
        results = []
        for probs in all_probs:
            scores = {name: round(float(p), 4) for name, p in zip(LABEL_NAMES, probs)}
            risk_level, primary_label = get_risk_level(scores)
            results.append({"scores": scores, "risk_level": risk_level, "primary_label": primary_label})
        return results


# ---------------------------------------------------------------------------
# Phrase highlighting helper (used by app.py)
# ---------------------------------------------------------------------------

def build_highlighted_html(text: str, word_weights: dict[str, float]) -> str:
    """
    Return HTML string where risk-contributing words are highlighted yellow
    and protective words are highlighted green.
    """
    if not word_weights:
        return f"<p>{text}</p>"

    tokens = re.split(r"(\s+)", text)
    parts = []
    for token in tokens:
        clean = re.sub(r"[^\w']", "", token.lower())
        weight = word_weights.get(clean, 0.0)
        if weight > 0.03:
            parts.append(
                f'<mark style="background:#FFD700;padding:1px 3px;border-radius:3px;'
                f'font-weight:600" title="risk signal: {weight:+.3f}">{token}</mark>'
            )
        elif weight < -0.03:
            parts.append(
                f'<mark style="background:#90EE90;padding:1px 3px;border-radius:3px" '
                f'title="protective signal: {weight:+.3f}">{token}</mark>'
            )
        else:
            parts.append(token)

    return '<p style="line-height:1.8">' + "".join(parts) + "</p>"
