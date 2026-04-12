"""
main.py — Federal Register daily bulk XML ingestion pipeline.

Usage:
    python main.py                        # ingest today's publication
    python main.py --date 2024-06-15      # ingest a specific date
    python main.py --date 2024-06-15 --prev 2024-06-14  # explicit previous date

Flow:
    1. Fetch today's (and yesterday's) bulk XML from the Federal Register API.
    2. Parse both XMLs into DocEntry maps.
    3. Diff consecutive publications at the token level.
    4. Classify each changed paragraph with DistilBERT (zero-shot NLI).
    5. Write structured records to PostgreSQL.
"""

import argparse
import hashlib
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

from classifier import classify_text
from db import init_schema, insert_changes, upsert_publication
from differ import diff_publications, parse_fedregister_xml

load_dotenv()

# ---------------------------------------------------------------------------
# Federal Register bulk XML endpoints
# ---------------------------------------------------------------------------

FR_BULK_URL = "https://www.federalregister.gov/api/v1/documents.json"
FR_FULL_TEXT_URL = "https://www.federalregister.gov/documents/full_text/xml/{date}.xml"
# Fallback: the official govinfo bulk data URL
GOVINFO_BULK = (
    "https://www.govinfo.gov/content/pkg/FR-{date}/xml/FR-{date}.xml"
)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)


def _cache_path(pub_date: date) -> Path:
    return DATA_DIR / f"fr_{pub_date.isoformat()}.xml"


def fetch_xml(pub_date: date) -> bytes:
    """
    Download or return cached bulk XML for a given publication date.
    Tries the Federal Register full-text endpoint first, then GovInfo.
    """
    cache = _cache_path(pub_date)
    if cache.exists():
        print(f"[fetch] Using cached XML: {cache}")
        return cache.read_bytes()

    date_str = pub_date.isoformat()
    urls = [
        FR_FULL_TEXT_URL.format(date=date_str),
        GOVINFO_BULK.format(date=date_str.replace("-", "")),
    ]
    for url in urls:
        print(f"[fetch] Trying {url} …")
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "reg-change-engine/1.0"})
            if r.status_code == 200 and r.content:
                cache.write_bytes(r.content)
                print(f"[fetch] Saved {len(r.content):,} bytes → {cache}")
                return r.content
            print(f"[fetch] HTTP {r.status_code} from {url}")
        except requests.RequestException as e:
            print(f"[fetch] Request error: {e}")

    raise RuntimeError(f"Could not fetch Federal Register XML for {pub_date}")


# ---------------------------------------------------------------------------
# Last business day helper
# ---------------------------------------------------------------------------

def prev_business_day(d: date) -> date:
    """Return the most recent weekday before d (skip Sat/Sun)."""
    delta = 1
    prev = d - timedelta(days=delta)
    while prev.weekday() >= 5:  # 5=Sat, 6=Sun
        delta += 1
        prev = d - timedelta(days=delta)
    return prev


# ---------------------------------------------------------------------------
# Main ingestion pipeline
# ---------------------------------------------------------------------------

def run(target_date: date, previous_date: date | None = None):
    if previous_date is None:
        previous_date = prev_business_day(target_date)

    print(f"\n{'='*60}")
    print(f"  reg-change-engine  |  {previous_date} → {target_date}")
    print(f"{'='*60}\n")

    # 1. Ensure DB schema exists
    init_schema()

    # 2. Fetch XML for both dates
    t0 = time.perf_counter()
    new_xml = fetch_xml(target_date)
    old_xml = fetch_xml(previous_date)
    print(f"[timing] Fetch: {time.perf_counter()-t0:.1f}s")

    # 3. Parse XML
    t0 = time.perf_counter()
    new_entries = parse_fedregister_xml(new_xml)
    old_entries = parse_fedregister_xml(old_xml)
    print(f"[parse] {len(old_entries)} old docs / {len(new_entries)} new docs")
    print(f"[timing] Parse: {time.perf_counter()-t0:.1f}s")

    # 4. Diff
    t0 = time.perf_counter()
    diffs = diff_publications(old_entries, new_entries)
    print(f"[diff] {len(diffs)} changed paragraphs")
    print(f"[timing] Diff: {time.perf_counter()-t0:.1f}s")

    if not diffs:
        print("[pipeline] No changes detected — nothing to write.")
        return

    # 5. Classify
    t0 = time.perf_counter()
    records = []
    for i, diff in enumerate(diffs):
        text_to_classify = diff.new_text or diff.old_text or ""
        try:
            clf = classify_text(text_to_classify)
            domain = clf.domain
            domain_score = clf.score
        except Exception as e:
            print(f"[classify] Error on record {i}: {e}")
            domain = "other"
            domain_score = 0.0

        records.append({
            "pub_date": target_date,
            "paragraph_hash": diff.paragraph_hash,
            "diff_type": diff.diff_type,
            "old_text": diff.old_text,
            "new_text": diff.new_text,
            "domain": domain,
            "domain_score": domain_score,
            "document_number": diff.document_number,
            "agency": diff.agency,
        })

        if (i + 1) % 50 == 0:
            print(f"  classified {i+1}/{len(diffs)} …")

    print(f"[timing] Classify: {time.perf_counter()-t0:.1f}s")

    # 6. Write to DB
    upsert_publication(target_date, raw_path=str(_cache_path(target_date)))
    insert_changes(records)

    # Summary
    from collections import Counter
    domain_counts = Counter(r["domain"] for r in records)
    print("\n[summary] Domain breakdown:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain:20s}: {count}")
    print(f"\n[done] {len(records)} records written for {target_date}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest and diff Federal Register XML publications."
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=date.today(),
        help="Target publication date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--prev",
        type=date.fromisoformat,
        default=None,
        help="Previous publication date to diff against. Defaults to last business day.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(target_date=args.date, previous_date=args.prev)
