-- Executed automatically by the official postgres entrypoint on first boot
-- against an EMPTY data directory (docker-entrypoint-initdb.d convention).
-- If the `postgres_data` volume already exists, this file is skipped, so
-- app/database.py also issues this statement defensively on startup.
CREATE EXTENSION IF NOT EXISTS vector;
