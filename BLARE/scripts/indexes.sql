-- Enable pg_trgm and build baseline indexes for BLARE-PG.
-- Replace <table> and <column> with your target log table and text column.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Case-sensitive path (default when regex is case-sensitive)
CREATE INDEX IF NOT EXISTS idx_logs_message_trgm
    ON <table> USING GIN (<column> gin_trgm_ops);

-- Case-insensitive path (used when regex flags imply (?i) or unknown)
CREATE INDEX IF NOT EXISTS idx_logs_message_trgm_lower
    ON <table> USING GIN (lower(<column>) gin_trgm_ops);

-- Optional: analyze to refresh planner stats.
ANALYZE <table>;
