"""
ingestion.py — Download and ingest Federal Register bulk XML publications.

Fetches daily FR bulk XML from the Government Publishing Office, parses
document entries via differ.py, classifies each changed paragraph via
classifier.py, and persists results to PostgreSQL via db.py.

Note: SEC fair-use policy requires a descriptive User-Agent header with
contact info. Set EDGAR_USER_AGENT in .env.

Usage:
    python ingestion.py --date 2026-04-11
    python ingestion.py --date 2026-04-11 --date 2026-04-10
    python ingestion.py --since 2026-04-01          # backfill a date range
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from dotenv import load_dotenv

import db
from differ import diff_publications, parse_fedregister_xml

load_dotenv()

FR_USER_AGENT = os.getenv(
    "FR_USER_AGENT",
    "reg-change-engine research@example.com",
)

# GPO bulk data: https://www.govinfo.gov/bulkdata/FR/{year}/{month}/
# Each file is a daily Federal Register XML archive (~40–80 MB).
GPO_BULK_URL = "https://www.govinfo.gov/bulkdata/FR/{year}/{month:02d}/FR-{date}.xml"

RATE_LIMIT_DELAY = 1.0   # GPO fair-use: be conservative, 1 req/s


def _headers() -> dict:
    return {
        "User-Agent": FR_USER_AGENT,
        "Accept": "application/xml, text/xml",
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def fetch_publication_xml(pub_date: date) -> Optional[bytes]:
    """
    Download the Federal Register bulk XML for a given publication date.
    Returns raw bytes, or None on failure.
    """
    url = GPO_BULK_URL.format(
        year=pub_date.year,
        month=pub_date.month,
        date=pub_date.isoformat(),
    )
    time.sleep(RATE_LIMIT_DELAY)
    try:
        print(f"[fetch] GET {url}")
        r = requests.get(url, headers=_headers(), timeout=120)
        if r.status_code == 200:
            print(f"[fetch] {len(r.content) // 1024} KB received")
            return r.content
        print(f"[fetch] HTTP {r.status_code} for {pub_date}")
    except requests.RequestException as e:
        print(f"[fetch] Request error: {e}")
    return None


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_date(pub_date: date):
    """
    Full pipeline for a single publication date:
      1. Fetch current and previous day XML.
      2. Parse both into DocEntry maps.
      3. Diff to produce DiffRecords.
      4. Classify each changed paragraph.
      5. Persist to PostgreSQL.
    """
    from classifier import classify_batch, DOMAIN_LABELS

    print(f"\n{'='*52}")
    print(f"  Ingesting Federal Register: {pub_date}")
    print(f"{'='*52}")

    # Load previous publication from DB if available, else try to fetch it
    prev_date = prev_business_day(pub_date)
    prev_xml = fetch_publication_xml(prev_date)
    curr_xml = fetch_publication_xml(pub_date)

    if curr_xml is None:
        print(f"[skip] No XML for {pub_date}")
        return

    curr_entries = parse_fedregister_xml(curr_xml)
    prev_entries = parse_fedregister_xml(prev_xml) if prev_xml else {}

    print(f"[diff] {len(prev_entries)} prev docs, {len(curr_entries)} curr docs")

    from differ import diff_publications
    diffs = diff_publications(prev_entries, curr_entries)
    print(f"[diff] {len(diffs)} changed paragraphs")

    if not diffs:
        db.upsert_publication(pub_date)
        return

    # Classify in batches
    texts = [d.new_text or d.old_text or "" for d in diffs]
    classifications = classify_batch(texts)

    records = []
    for diff, clf in zip(diffs, classifications):
        records.append({
            "pub_date": pub_date,
            "paragraph_hash": diff.paragraph_hash,
            "diff_type": diff.diff_type,
            "old_text": diff.old_text,
            "new_text": diff.new_text,
            "domain": clf.domain,
            "domain_score": clf.score,
            "document_number": diff.document_number,
            "agency": diff.agency,
        })

    db.upsert_publication(pub_date)
    db.insert_changes(records)
    print(f"[done] {len(records)} records persisted for {pub_date}")


def prev_business_day(d: date) -> date:
    """Return the most recent business day before d (skips weekends)."""
    candidate = d - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ingest Federal Register bulk XML for one or more publication dates."
    )
    parser.add_argument(
        "--date", dest="dates", action="append", type=date.fromisoformat,
        metavar="YYYY-MM-DD", help="Publication date to ingest (repeatable)",
    )
    parser.add_argument(
        "--since", type=date.fromisoformat, metavar="YYYY-MM-DD",
        help="Backfill all business days from this date through today",
    )
    args = parser.parse_args()

    db.init_schema()

    targets: list[date] = []
    if args.since:
        d = args.since
        today = date.today()
        while d <= today:
            if d.weekday() < 5:
                targets.append(d)
            d += timedelta(days=1)
    if args.dates:
        targets.extend(args.dates)
    if not targets:
        targets = [date.today()]

    for pub_date in sorted(set(targets)):
        ingest_date(pub_date)

    print("\n[ingestion] Done.")


if __name__ == "__main__":
    main()
