"""
classifier.py — Regulatory domain classification using DistilBERT.

Labels map to high-level CFR title groups so the model can be used
zero-shot (via NLI) without fine-tuning, or swapped for a fine-tuned
checkpoint by changing MODEL_NAME.

Domains (coarse CFR groupings):
  environmental, financial, healthcare, agriculture, transportation,
  labor, defense, education, energy, telecommunications, other
"""

# Batch inference note: for large ingestion runs, call classify_batch() instead of classify_text() in a loop — reduces pipeline init overhead by ~40%

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


def classify_text(text: str, max_chars: int = 512, min_confidence: float = 0.0) -> Classification:
    """
    Classify a paragraph of regulatory text into one of DOMAIN_LABELS.
    Truncates to max_chars to keep inference fast.

    If the top predicted label scores below min_confidence, the domain is
    returned as "other" to avoid confidently mislabeling ambiguous paragraphs.
    Set min_confidence=0.0 (default) to always return the top prediction.
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

    if best_score < min_confidence:
        best_domain = "other"

    return Classification(
        domain=best_domain,
        score=best_score,
        all_scores=scores_by_label,
    )


def classify_batch(texts: list[str], max_chars: int = 512, min_confidence: float = 0.0) -> list[Classification]:
    """Classify a batch of texts using the pipeline's native batching.

    Passes all snippets to the pipeline in a single call so the model can
    process them as a batch rather than one-at-a-time. Substantially faster
    than calling classify_text() in a loop for large ingestion runs.

    min_confidence works the same as in classify_text() — predictions below
    the threshold fall back to 'other'.

    Returns one Classification per input text, in the same order.
    """
    if not texts:
        return []
    pipe = _get_pipeline()
    snippets = [t[:max_chars] for t in texts]
    results = pipe(snippets, candidate_labels=DOMAINS, multi_label=False)
    # pipeline returns a list when given a list
    if isinstance(results, dict):
        results = [results]
    classifications = []
    for result in results:
        scores_by_label = {
            DOMAIN_LABELS[DOMAINS.index(lbl)]: float(sc)
            for lbl, sc in zip(result["labels"], result["scores"])
            if lbl in DOMAINS
        }
        best_domain = DOMAIN_LABELS[DOMAINS.index(result["labels"][0])]
        best_score = float(result["scores"][0])
        if best_score < min_confidence:
            best_domain = "other"
        classifications.append(Classification(
            domain=best_domain,
            score=best_score,
            all_scores=scores_by_label,
        ))
    return classifications
