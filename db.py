"""
db.py — PostgreSQL schema setup and write helpers for reg-change-engine.
"""

import os
import psycopg2
import psycopg2.extras
from datetime import date
from dotenv import load_dotenv

load_dotenv()

DSN = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/regchanges")


def get_conn():
    return psycopg2.connect(DSN)


DDL = """
CREATE TABLE IF NOT EXISTS publications (
    id          SERIAL PRIMARY KEY,
    pub_date    DATE        NOT NULL UNIQUE,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_path    TEXT
);

CREATE TABLE IF NOT EXISTS changed_paragraphs (
    id              SERIAL PRIMARY KEY,
    pub_date        DATE        NOT NULL REFERENCES publications(pub_date),
    paragraph_hash  TEXT        NOT NULL,
    diff_type       TEXT        NOT NULL CHECK (diff_type IN ('added','removed','modified')),
    old_text        TEXT,
    new_text        TEXT,
    domain          TEXT,          -- predicted regulatory domain
    domain_score    REAL,          -- confidence
    document_number TEXT,
    agency          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cp_pub_date ON changed_paragraphs(pub_date);
CREATE INDEX IF NOT EXISTS idx_cp_domain   ON changed_paragraphs(domain);
"""


def init_schema():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
    print("[db] Schema initialised.")


def upsert_publication(pub_date: date, raw_path: str | None = None):
    sql = """
    INSERT INTO publications (pub_date, raw_path)
    VALUES (%s, %s)
    ON CONFLICT (pub_date) DO UPDATE SET raw_path = EXCLUDED.raw_path
    RETURNING id
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (pub_date, raw_path))
            row = cur.fetchone()
        conn.commit()
    return row[0] if row else None


def insert_changes(records: list[dict]):
    """Bulk-insert changed paragraph records."""
    if not records:
        return
    sql = """
    INSERT INTO changed_paragraphs
        (pub_date, paragraph_hash, diff_type, old_text, new_text,
         domain, domain_score, document_number, agency)
    VALUES
        (%(pub_date)s, %(paragraph_hash)s, %(diff_type)s, %(old_text)s, %(new_text)s,
         %(domain)s, %(domain_score)s, %(document_number)s, %(agency)s)
    """
    with get_conn() as conn:
        psycopg2.extras.execute_batch(conn.cursor(), sql, records)
        conn.commit()
    print(f"[db] Inserted {len(records)} change records.")


def query_changes(
    pub_date: date | None = None,
    domain: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    conditions = []
    params: list = []
    if pub_date:
        conditions.append("pub_date = %s")
        params.append(pub_date)
    if domain:
        conditions.append("domain = %s")
        params.append(domain)
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"""
    SELECT id, pub_date, diff_type, old_text, new_text,
           domain, domain_score, document_number, agency, created_at
    FROM changed_paragraphs
    {where}
    ORDER BY created_at DESC
    LIMIT %s OFFSET %s
    """
    params += [limit, offset]
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
