CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS docs (
    id BIGSERIAL PRIMARY KEY,
    text_content TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS free_index_meta (
    index_id TEXT PRIMARY KEY,
    c DOUBLE PRECISION NOT NULL,
    lmax INTEGER NOT NULL,
    use_shell BOOLEAN NOT NULL,
    discovery_rows BIGINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    postings_codec TEXT NOT NULL DEFAULT 'zlib_delta',
    postings_params JSONB,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS free_keys (
    index_id TEXT NOT NULL,
    gram TEXT NOT NULL,
    df_discovery INTEGER NOT NULL,
    sel_discovery DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (index_id, gram)
);

CREATE TABLE IF NOT EXISTS free_index (
    index_id TEXT NOT NULL,
    gram TEXT NOT NULL,
    postings BYTEA NOT NULL,
    df_full INTEGER NOT NULL,
    sel_full DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (index_id, gram)
);

CREATE TABLE IF NOT EXISTS free_meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
