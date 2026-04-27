# reg-change-engine

**[Live Demo →](https://astew24.github.io/reg-change-engine/)**

## What this does

`reg-change-engine` is a production-grade Python pipeline that automatically **tracks daily changes to U.S. federal regulations** by:

1. **Fetching** the Federal Register's official bulk XML publication each business day
2. **Diffing** consecutive publications at the token level — surfacing exactly which words and paragraphs changed, were added, or were removed
3. **Classifying** every changed paragraph into a regulatory domain (environmental, financial, healthcare, etc.) using a zero-shot DistilBERT NLI model
4. **Storing** structured change records in PostgreSQL with full provenance
5. **Serving** the results through a FastAPI REST API with filtering by date and domain

Useful for compliance teams, policy researchers, legal-tech products, or anyone who needs to know *exactly what changed in federal regulations today*.

---

## Project structure

```
reg-change-engine/
├── main.py          # Ingestion pipeline (fetch → diff → classify → write)
├── differ.py        # Token-level XML diff engine
├── classifier.py    # DistilBERT zero-shot regulatory domain classifier
├── db.py            # PostgreSQL schema, write helpers, query helpers
├── app.py           # FastAPI app — /changes, /domains, /health endpoints
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick start

### 1. Clone and set up environment

```bash
git clone <repo>
cd reg-change-engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Start Postgres

```bash
docker-compose up -d postgres
```

### 3. Run ingestion

```bash
# Ingest today vs yesterday
python main.py

# Ingest a specific date
python main.py --date 2024-06-15

# Specify both dates explicitly
python main.py --date 2024-06-15 --prev 2024-06-14
```

The first run downloads and caches the XML files under `data/`, initialises the schema, and writes change records.

### 4. Start the API

```bash
uvicorn app:app --reload
# → http://localhost:8000/docs  (Swagger UI)
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/changes` | Paginated list of changed paragraphs. Filter by `pub_date` and `domain`. |
| `GET` | `/changes/{id}` | Single change record |
| `GET` | `/domains` | Count of changes per regulatory domain |
| `GET` | `/stats` | Total changes + avg classifier confidence grouped by domain |
| `GET` | `/publications` | List of ingested publication dates |
| `GET` | `/health` | Liveness + DB connectivity check |

Example:

```
GET /changes?pub_date=2024-06-15&domain=environmental&limit=20
```

---

## Database schema

```sql
publications (id, pub_date UNIQUE, fetched_at, raw_path)

changed_paragraphs (
  id, pub_date → publications,
  paragraph_hash,
  diff_type IN ('added','removed','modified'),
  old_text, new_text,
  domain,           -- e.g. 'environmental'
  domain_score,     -- classifier confidence 0–1
  document_number, agency, created_at
)
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/regchanges` | Postgres DSN |
| `DATA_DIR` | `data` | Local cache for downloaded XML files |
| `CLASSIFIER_MODEL` | `cross-encoder/nli-distilroberta-base` | HuggingFace model ID |

---

## Design decisions

- **Token-level diff** uses Python's `difflib.SequenceMatcher` for word-boundary tokenisation — no external diff library needed.
- **Zero-shot NLI** means the classifier works out-of-the-box without labelled regulatory training data. Swap `CLASSIFIER_MODEL` for a fine-tuned checkpoint to boost accuracy.
- **Bulk XML** (not the JSON API) is used because it contains the full text of every document in one atomic download, making diffing reliable.
- **Caching** raw XML to disk ensures reruns don't re-fetch, and preserves the exact snapshot for auditing.
- **Batch inference** in `classify_batch()` passes all snippets to the HuggingFace pipeline in one call — meaningfully faster than looping `classify_text()` per paragraph on large ingestion runs.
- **Min-confidence fallback**: predictions below the configurable `min_confidence` threshold are returned as `"other"` so the API doesn't surface confidently-mislabelled domains for ambiguous text.

---

## Running end-to-end with Docker

```bash
docker-compose up --build
# Postgres on :5432, API on :8000
docker-compose exec api python main.py
```
