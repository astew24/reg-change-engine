"""
app.py — FastAPI service exposing regulatory change records.

Endpoints:
  GET /changes         - paginated list of changed paragraphs
  GET /changes/{id}    - single record
  GET /domains         - distinct domain labels + counts
  GET /health          - liveness check
"""

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

import db

app = FastAPI(
    title="Reg Change Engine API",
    description="Surfaces daily token-level diffs of Federal Register publications, "
                "classified by regulatory domain via DistilBERT.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChangeRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    pub_date: date
    diff_type: str
    old_text: Optional[str]
    new_text: Optional[str]
    domain: Optional[str]
    domain_score: Optional[float]
    document_number: Optional[str]
    agency: Optional[str]


class ChangesResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[ChangeRecord]


class DomainCount(BaseModel):
    domain: str
    count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    """Liveness check."""
    try:
        conn = db.get_conn()
        conn.close()
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "degraded", "db": str(e)}


@app.get("/changes", response_model=ChangesResponse, tags=["changes"])
def list_changes(
    pub_date: Optional[date] = Query(None, description="Filter by publication date (YYYY-MM-DD)"),
    domain: Optional[str] = Query(None, description="Filter by regulatory domain"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """
    Return paginated regulatory change records.

    Optionally filter by `pub_date` and/or `domain`.
    """
    rows = db.query_changes(pub_date=pub_date, domain=domain, limit=limit, offset=offset)

    # Count total (simple approach — re-query without limit)
    all_rows = db.query_changes(pub_date=pub_date, domain=domain, limit=100_000, offset=0)

    return ChangesResponse(
        total=len(all_rows),
        limit=limit,
        offset=offset,
        items=[ChangeRecord(**r) for r in rows],
    )


@app.get("/changes/{record_id}", response_model=ChangeRecord, tags=["changes"])
def get_change(record_id: int):
    """Fetch a single change record by ID."""
    sql = """
    SELECT id, pub_date, diff_type, old_text, new_text,
           domain, domain_score, document_number, agency, created_at
    FROM changed_paragraphs WHERE id = %s
    """
    import psycopg2.extras
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, (record_id,))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found")
    return ChangeRecord(**dict(row))


@app.get("/domains", response_model=list[DomainCount], tags=["analytics"])
def list_domains(
    pub_date: Optional[date] = Query(None, description="Scope to a specific publication date"),
):
    """Return counts of changed paragraphs by regulatory domain."""
    where = "WHERE pub_date = %(pub_date)s" if pub_date else ""
    sql = f"""
    SELECT domain, COUNT(*) AS count
    FROM changed_paragraphs
    {where}
    GROUP BY domain
    ORDER BY count DESC
    """
    import psycopg2.extras
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, {"pub_date": pub_date} if pub_date else {})
            return [DomainCount(**dict(r)) for r in cur.fetchall()]


@app.get("/stats", tags=["analytics"])
def stats_summary():
    """Return total changes grouped by domain and date."""
    sql = """
    SELECT domain, COUNT(*) AS total_changes, AVG(domain_score) AS avg_score
    FROM changed_paragraphs
    GROUP BY domain
    ORDER BY total_changes DESC
    """
    import psycopg2.extras
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]


@app.get("/publications", tags=["meta"])
def list_publications():
    """Return dates of all ingested publications."""
    sql = "SELECT pub_date, fetched_at, raw_path FROM publications ORDER BY pub_date DESC"
    import psycopg2.extras
    with db.get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Dev server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
