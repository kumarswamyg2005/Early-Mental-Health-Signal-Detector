"""
BERT + LIWC-style linguistic features for mental health signal detection.
Supports save_pretrained() / load via HuggingFace conventions.
"""

import os
import torch
import torch.nn as nn
from transformers import BertModel

LABEL_NAMES = ["depression", "anxiety", "crisis", "neutral"]
NUM_LABELS = len(LABEL_NAMES)
NUM_LING_FEATURES = 8


class MentalHealthClassifier(nn.Module):
    """BERT CLS embedding + projected linguistic features → 4-class classifier."""

    LABELS = LABEL_NAMES

    def __init__(self, num_ling_features: int = NUM_LING_FEATURES):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        bert_hidden = self.bert.config.hidden_size  # 768

        self.ling_proj = nn.Sequential(
            nn.Linear(num_ling_features, 32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden + 32, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_LABELS),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ling_features: torch.Tensor,
    ) -> torch.Tensor:
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = bert_out.last_hidden_state[:, 0]          # [B, 768]
        ling = self.ling_proj(ling_features)             # [B, 32]
        return self.classifier(torch.cat([cls, ling], dim=1))

    # ------------------------------------------------------------------
    # Persistence (save_pretrained style for BERT + torch.save for head)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.bert.save_pretrained(os.path.join(path, "bert"))
        torch.save(
            {
                "ling_proj": self.ling_proj.state_dict(),
                "classifier": self.classifier.state_dict(),
            },
            os.path.join(path, "head.pt"),
        )

    @classmethod
    def load(cls, path: str, num_ling_features: int = NUM_LING_FEATURES) -> "MentalHealthClassifier":
        model = cls(num_ling_features=num_ling_features)
        model.bert = BertModel.from_pretrained(os.path.join(path, "bert"))
        head = torch.load(os.path.join(path, "head.pt"), map_location="cpu")
        model.ling_proj.load_state_dict(head["ling_proj"])
        model.classifier.load_state_dict(head["classifier"])
        return model
