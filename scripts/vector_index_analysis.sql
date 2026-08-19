-- =============================================================================
-- Vector index execution analysis: HNSW vs. IVFFlat vs. sequential scan.
--
-- Run after `python scripts/seed_tickets.py` (or after the API has triaged
-- a few real tickets) so `tickets.embedding` actually holds rows to search.
--
-- Usage (from the project root, once `docker compose up -d` is running):
--
--   docker exec -i triage_postgres psql -U triage_user -d triage_engine \
--     -v ON_ERROR_STOP=1 -f - < scripts/vector_index_analysis.sql
--
-- or open an interactive session and paste sections one at a time:
--
--   docker exec -it triage_postgres psql -U triage_user -d triage_engine
--
-- Read this top to bottom -- later sections build on state (a temp IVFFlat
-- index, session GUCs) created by earlier ones.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 0. Sanity check: row count and the index app/database.py's init_models()
--    already created (HNSW, cosine ops -- see the _HNSW_INDEX_DDL comment
--    there for why HNSW is the default over IVFFlat).
-- -----------------------------------------------------------------------------

SELECT count(*) AS total_tickets, count(embedding) AS embedded_tickets FROM tickets;

SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'tickets';


-- -----------------------------------------------------------------------------
-- 1. EXPLAIN ANALYZE a real similarity query using the existing HNSW index.
--
-- The query shape below is exactly what app/services/rag.py's
-- `find_similar_tickets` issues: ORDER BY the `<=>` (cosine distance)
-- expression, LIMIT k. This is the ONLY shape that lets the planner use
-- the vector index -- ORDER BY on a derived/negated expression, or a
-- WHERE-clause distance filter without a matching ORDER BY, falls back to
-- a full sequential scan.
--
-- Look for "Index Scan using tickets_embedding_hnsw_cosine_idx" in the
-- plan below. If you instead see "Seq Scan on tickets", something (an
-- unindexed column type, a query-shape mismatch, or a corpus too small
-- for the planner to bother with the index) is preventing index usage.
-- -----------------------------------------------------------------------------

EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT id, title, extracted_error, resolution,
       1 - (embedding <=> (SELECT embedding FROM tickets WHERE is_verified LIMIT 1)) AS similarity
FROM tickets
WHERE embedding IS NOT NULL
  AND resolution IS NOT NULL
  AND is_verified = true
ORDER BY embedding <=> (SELECT embedding FROM tickets WHERE is_verified LIMIT 1)
LIMIT 3;


-- -----------------------------------------------------------------------------
-- 2. `hnsw.ef_search`: the accuracy/speed knob at query time.
--
-- Higher ef_search explores more of the HNSW graph per query -- better
-- recall, more latency. Default is 40. Bump it for small/noisy corpora
-- (like a freshly-seeded 8-row table, where the graph is nearly flat and
-- the default may already visit everything), lower it once the corpus is
-- large and query latency starts to matter more than the last percent of
-- recall.
-- -----------------------------------------------------------------------------

SET hnsw.ef_search = 100;

EXPLAIN (ANALYZE, BUFFERS)
SELECT id, extracted_error
FROM tickets
WHERE embedding IS NOT NULL
ORDER BY embedding <=> (SELECT embedding FROM tickets WHERE is_verified LIMIT 1)
LIMIT 3;

RESET hnsw.ef_search;


-- -----------------------------------------------------------------------------
-- 3. Force a sequential scan for comparison -- this is what EVERY query
--    would look like without the vector index, and what a query-shape
--    mistake (see the comment in section 1) silently degrades to.
--
-- On a corpus this small the cost difference will look negligible or
-- even favor the seq scan (the planner is right to do that below a size
-- threshold -- an index has overhead a tiny table doesn't justify). Re-run
-- this section after `python scripts/seed_tickets.py` has run several
-- times against a larger synthetic corpus (thousands of rows) to see the
-- planner switch, and the cost/timing gap actually widen.
-- -----------------------------------------------------------------------------

SET enable_indexscan = off;
SET enable_bitmapscan = off;

EXPLAIN (ANALYZE, BUFFERS)
SELECT id, extracted_error
FROM tickets
WHERE embedding IS NOT NULL
ORDER BY embedding <=> (SELECT embedding FROM tickets WHERE is_verified LIMIT 1)
LIMIT 3;

RESET enable_indexscan;
RESET enable_bitmapscan;


-- -----------------------------------------------------------------------------
-- 4. IVFFlat as an alternative: build it, ANALYZE, then compare.
--
-- Unlike HNSW (a graph index, built incrementally, correct on an empty
-- table), IVFFlat clusters existing rows into `lists` partitions at BUILD
-- time -- it needs representative data present *before* creation, and an
-- ANALYZE afterward so the planner has accurate statistics on it. Building
-- it on an empty or near-empty table (as this demo corpus is) produces a
-- low-quality index; the standard guidance is `lists = rows / 1000` for
-- corpora over ~10K rows (fewer lists below that).
--
-- This creates a SECOND index alongside the HNSW one purely for this
-- comparison -- drop it afterward (section 5) so production traffic keeps
-- using HNSW, which is the better default per app/database.py's comment.
-- -----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS tickets_embedding_ivfflat_cosine_idx
ON tickets USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);

ANALYZE tickets;

-- ivfflat.probes: how many of the `lists` partitions to search per query.
-- More probes = better recall, closer to (but never cheaper than) a full
-- scan; default is 1, which is fast but can miss neighbors sitting in a
-- partition that wasn't probed.
SET ivfflat.probes = 5;

-- Force the planner to prefer IVFFlat over the HNSW index that also
-- matches this query, purely so this EXPLAIN demonstrates IVFFlat's plan
-- shape ("Index Scan using tickets_embedding_ivfflat_cosine_idx").
-- In production, never disable indexes like this -- let the planner choose.
SET enable_seqscan = off;
BEGIN;
  DROP INDEX IF EXISTS tickets_embedding_hnsw_cosine_idx;
  EXPLAIN (ANALYZE, BUFFERS)
  SELECT id, extracted_error
  FROM tickets
  WHERE embedding IS NOT NULL
  ORDER BY embedding <=> (SELECT embedding FROM tickets WHERE is_verified LIMIT 1)
  LIMIT 3;
ROLLBACK;  -- restores the HNSW index; nothing above this line persists
RESET enable_seqscan;
RESET ivfflat.probes;


-- -----------------------------------------------------------------------------
-- 5. Cleanup: drop the comparison IVFFlat index. HNSW remains the
--    production index (it was only hidden inside the rolled-back
--    transaction in section 4, never actually dropped).
-- -----------------------------------------------------------------------------

DROP INDEX IF EXISTS tickets_embedding_ivfflat_cosine_idx;

SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'tickets';
