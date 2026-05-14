"""
FastAPI backend for the Mental Health Signal Detector.
Wraps the existing inference pipeline with a REST API.

No input text is stored. All inference is local.
"""

import sys
import os

# Allow imports from project root (pipeline/, models/)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from pipeline.analyze import MentalHealthAnalyzer, RISK_RESOURCES

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mental Health Signal Detector API",
    description="Local inference API — no text is stored or transmitted externally.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded analyzer
_analyzer: Optional[MentalHealthAnalyzer] = None

def get_analyzer() -> MentalHealthAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = MentalHealthAnalyzer(model_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved"))
    return _analyzer


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    num_lime_samples: int = Field(default=50, ge=10, le=200)

class BatchAnalyzeRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=50)

class ResourceItem(BaseModel):
    name: str
    detail: str

# top_notable_features() returns List[Dict[str, str]] with keys: feature, description, value
class TopFeature(BaseModel):
    feature: str
    description: str
    value: str

class AnalyzeResponse(BaseModel):
    risk_level: str
    primary_label: str
    scores: dict[str, float]
    word_weights: dict[str, float]
    top_features: list[TopFeature]
    feature_values: dict[str, float]   # describe_features() returns Dict[str, float]
    resources: list[ResourceItem]

class BatchResult(BaseModel):
    index: int
    risk_level: str
    primary_label: str
    scores: dict[str, float]

class BatchAnalyzeResponse(BaseModel):
    results: list[BatchResult]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _analyzer is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        analyzer = get_analyzer()
        result = analyzer.analyze(req.text.strip(), num_lime_samples=req.num_lime_samples)
        return AnalyzeResponse(
            risk_level=result["risk_level"],
            primary_label=result["primary_label"],
            scores=result["scores"],
            word_weights=result["word_weights"],
            top_features=result["top_features"],
            feature_values=result["feature_values"],
            resources=[ResourceItem(**r) for r in result["resources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
def analyze_batch(req: BatchAnalyzeRequest):
    try:
        analyzer = get_analyzer()
        cleaned = [t.strip() for t in req.texts if t.strip()]
        results = analyzer.analyze_batch(cleaned)
        return BatchAnalyzeResponse(
            results=[
                BatchResult(
                    index=i,
                    risk_level=r["risk_level"],
                    primary_label=r["primary_label"],
                    scores=r["scores"],
                )
                for i, r in enumerate(results)
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
