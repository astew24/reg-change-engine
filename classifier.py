"""
classifier.py — Regulatory domain classification using DistilBERT.

Labels map to high-level CFR title groups so the model can be used
zero-shot (via NLI) without fine-tuning, or swapped for a fine-tuned
checkpoint by changing MODEL_NAME.

Domains (coarse CFR groupings):
  environmental, financial, healthcare, agriculture, transportation,
  labor, defense, education, energy, telecommunications, other
"""

import os
import functools
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv(
    "CLASSIFIER_MODEL",
    "cross-encoder/nli-distilroberta-base",  # zero-shot NLI head
)

DOMAINS = [
    "environmental regulation",
    "financial regulation and banking",
    "healthcare and public health",
    "agriculture and food safety",
    "transportation and infrastructure",
    "labor and employment",
    "defense and national security",
    "education policy",
    "energy and natural resources",
    "telecommunications and media",
    "other government regulation",
]

# Short label used in the DB
DOMAIN_LABELS = [
    "environmental",
    "financial",
    "healthcare",
    "agriculture",
    "transportation",
    "labor",
    "defense",
    "education",
    "energy",
    "telecommunications",
    "other",
]


@dataclass
class Classification:
    domain: str
    score: float
    all_scores: dict[str, float]


@functools.lru_cache(maxsize=1)
def _get_pipeline():
    """Lazy-load the HuggingFace zero-shot classification pipeline."""
    from transformers import pipeline  # type: ignore
    print(f"[classifier] Loading model: {MODEL_NAME} …")
    return pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=-1,   # CPU; set to 0 for GPU
    )


def classify_text(text: str, max_chars: int = 512) -> Classification:
    """
    Classify a paragraph of regulatory text into one of DOMAIN_LABELS.
    Truncates to max_chars to keep inference fast.
    """
    pipe = _get_pipeline()
    snippet = text[:max_chars]
    result = pipe(snippet, candidate_labels=DOMAINS, multi_label=False)

    # result["labels"] / result["scores"] are sorted by descending score
    scores_by_label = {
        DOMAIN_LABELS[DOMAINS.index(lbl)]: float(sc)
        for lbl, sc in zip(result["labels"], result["scores"])
        if lbl in DOMAINS
    }
    best_domain = DOMAIN_LABELS[DOMAINS.index(result["labels"][0])]
    best_score = float(result["scores"][0])

    return Classification(
        domain=best_domain,
        score=best_score,
        all_scores=scores_by_label,
    )


def classify_batch(texts: list[str], max_chars: int = 512) -> list[Classification]:
    """Classify a batch of texts. Returns one Classification per text."""
    return [classify_text(t, max_chars) for t in texts]
