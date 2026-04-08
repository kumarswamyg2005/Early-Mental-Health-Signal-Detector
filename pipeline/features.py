"""
LIWC-style linguistic feature extraction.
All computation is local; no text leaves this process.
"""

import re
from typing import Dict, List
import numpy as np

# ---------------------------------------------------------------------------
# Word lists (inline — no external files required)
# ---------------------------------------------------------------------------

_FIRST_PERSON = frozenset([
    "i", "me", "my", "myself", "mine",
    "i'm", "i've", "i'll", "i'd", "i'd",
])

_NEGATIVE_WORDS = frozenset([
    # sadness / depression
    "sad", "unhappy", "miserable", "depressed", "hopeless", "worthless",
    "useless", "empty", "numb", "broken", "hurt", "pain", "suffer",
    "suffering", "terrible", "awful", "horrible", "dreadful", "despair",
    "despairing", "lonely", "alone", "isolated", "abandoned", "rejected",
    "failure", "failed", "loser", "hate", "hated",
    # anxiety
    "anxious", "worried", "nervous", "scared", "afraid", "fearful",
    "panic", "panicking", "terrified", "terror", "dread", "dreading",
    "stressed", "stress", "overwhelmed",
    # general negative
    "bad", "wrong", "worse", "worst", "dark", "heavy", "burden",
    "unbearable", "tired", "exhausted", "drained", "weak", "helpless",
    "powerless", "trapped", "stuck", "lost", "defeated", "devastated",
    "shattered", "crushed", "hollow", "void", "meaningless", "pointless",
])

_POSITIVE_WORDS = frozenset([
    "happy", "joy", "joyful", "glad", "pleased", "content", "peaceful",
    "calm", "grateful", "thankful", "hopeful", "hope", "love", "loved",
    "caring", "supported", "better", "improving", "good", "great",
    "wonderful", "amazing", "beautiful", "bright", "positive", "okay",
    "fine", "alright", "well", "healthy", "strong", "confident",
    "excited", "proud", "fulfilled", "connected", "understood",
])

_CRISIS_WORDS = frozenset([
    "suicide", "suicidal", "kill", "killing", "killed",
    "die", "dying", "death", "dead",
    "overdose", "selfharm", "self-harm", "cutting",
    "method", "plan", "goodbye", "farewell",
    "disappear", "vanish", "not worth living", "no reason to live",
    "better off dead", "end it", "end my life", "take my life",
    "want to die", "going to die", "won't be here",
])

_COGPROC_WORDS = frozenset([
    "think", "thinking", "thought", "know", "knowing", "understand",
    "consider", "because", "reason", "realize", "realized", "believe",
    "feel", "feeling", "felt", "sense", "seem", "seemed", "appear",
    "wonder", "wondering", "remember", "forget", "decide", "decided",
    "maybe", "perhaps", "possibly", "probably", "might",
])

FEATURE_NAMES = [
    "word_count_norm",
    "first_person_rate",
    "neg_word_ratio",
    "pos_word_ratio",
    "crisis_word_ratio",
    "cogproc_rate",
    "neg_pos_diff",
    "avg_sentence_length_norm",
]

FEATURE_LABELS = {
    "word_count_norm": "Text length",
    "first_person_rate": "First-person pronoun rate",
    "neg_word_ratio": "Negative language ratio",
    "pos_word_ratio": "Positive language ratio",
    "crisis_word_ratio": "Crisis vocabulary ratio",
    "cogproc_rate": "Cognitive processing words",
    "neg_pos_diff": "Net negative sentiment",
    "avg_sentence_length_norm": "Average sentence length",
}


def extract_features(text: str) -> np.ndarray:
    """Return an 8-dim float32 array of LIWC-style features."""
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    n = max(len(tokens), 1)
    n_sent = max(len(sentences), 1)

    word_count_norm = min(len(tokens) / 2000.0, 1.0)
    fp_rate = sum(1 for t in tokens if t in _FIRST_PERSON) / n
    neg_ratio = sum(1 for t in tokens if t in _NEGATIVE_WORDS) / n
    pos_ratio = sum(1 for t in tokens if t in _POSITIVE_WORDS) / n
    crisis_ratio = sum(1 for t in tokens if t in _CRISIS_WORDS) / n
    cogproc_rate = sum(1 for t in tokens if t in _COGPROC_WORDS) / n
    neg_pos_diff = (neg_ratio - pos_ratio + 1.0) / 2.0  # normalised [0,1]
    avg_sent_len_norm = min((len(tokens) / n_sent) / 50.0, 1.0)

    return np.array(
        [word_count_norm, fp_rate, neg_ratio, pos_ratio,
         crisis_ratio, cogproc_rate, neg_pos_diff, avg_sent_len_norm],
        dtype=np.float32,
    )


def describe_features(features: np.ndarray) -> Dict[str, float]:
    return dict(zip(FEATURE_NAMES, features.tolist()))


# Priority order for human-readable explanations (higher = explain first)
_FEATURE_PRIORITY = [
    "crisis_word_ratio",
    "neg_word_ratio",
    "first_person_rate",
    "neg_pos_diff",
    "pos_word_ratio",
    "cogproc_rate",
    "word_count_norm",
    "avg_sentence_length_norm",
]

_FEATURE_THRESHOLDS: Dict[str, float] = {
    "crisis_word_ratio": 0.005,
    "neg_word_ratio": 0.04,
    "first_person_rate": 0.06,
    "neg_pos_diff": 0.65,
    "pos_word_ratio": 0.03,
    "cogproc_rate": 0.04,
    "word_count_norm": 0.05,
    "avg_sentence_length_norm": 0.1,
}


def top_notable_features(features: np.ndarray) -> List[Dict[str, str]]:
    """Return up to 3 linguistically notable features with plain-English descriptions."""
    named = describe_features(features)
    results: List[Dict[str, str]] = []

    for key in _FEATURE_PRIORITY:
        val = named[key]
        threshold = _FEATURE_THRESHOLDS.get(key, 0.0)
        if val < threshold:
            continue

        pct = f"{val:.1%}"
        if key == "crisis_word_ratio":
            desc = f"Crisis-related vocabulary detected ({pct} of words)"
        elif key == "neg_word_ratio":
            desc = f"Elevated negative language ({pct} of words)"
        elif key == "first_person_rate":
            desc = f"High first-person pronoun use ({pct} of words)"
        elif key == "neg_pos_diff" and val > 0.65:
            desc = f"Text skews strongly negative (net sentiment score {val:.2f})"
        elif key == "pos_word_ratio":
            desc = f"Low positive language ({pct} of words)"
        elif key == "cogproc_rate":
            desc = f"Elevated cognitive-processing language ({pct} of words)"
        else:
            desc = f"{FEATURE_LABELS[key]}: {pct}"

        results.append({"feature": FEATURE_LABELS[key], "description": desc, "value": pct})
        if len(results) == 3:
            break

    if not results:
        results.append({
            "feature": "No strong linguistic signals",
            "description": "Text does not contain notable risk-related language patterns.",
            "value": "—",
        })

    return results
