-- Regex span helpers ensuring character-based offsets

CREATE FUNCTION regex_match_span(
    txt        text,
    pattern    text,
    start_pos  int,
    end_pos    int
) RETURNS boolean
AS 'MODULE_PATHNAME', 'regex_match_span'
LANGUAGE C
STRICT
STABLE
PARALLEL SAFE;

CREATE FUNCTION regex_match_from_upto(
    txt        text,
    pattern    text,
    start_pos  int,
    end_pos    int
) RETURNS boolean
AS 'MODULE_PATHNAME', 'regex_match_from_upto'
LANGUAGE C
STRICT
STABLE
PARALLEL SAFE;

CREATE FUNCTION regex_find_from(
    txt        text,
    pattern    text,
    start_pos  int,
    allow_empty boolean DEFAULT false
) RETURNS int[]
AS 'MODULE_PATHNAME', 'regex_find_from'
LANGUAGE C
STRICT
STABLE
PARALLEL SAFE;

CREATE FUNCTION regex_find_at_upto(
    txt        text,
    pattern    text,
    start_pos  int,
    end_pos    int,
    allow_empty boolean DEFAULT false
) RETURNS int[]
AS 'MODULE_PATHNAME', 'regex_find_at_upto'
LANGUAGE C
STRICT
STABLE
PARALLEL SAFE;

CREATE FUNCTION regex_find_all(
    txt        text,
    pattern    text,
    start_pos  int DEFAULT 1,
    overlap    boolean DEFAULT true,
    allow_empty boolean DEFAULT false
) RETURNS SETOF int[]
AS 'MODULE_PATHNAME', 'regex_find_all'
LANGUAGE C
STRICT
STABLE
PARALLEL SAFE;

CREATE FUNCTION regex_verify_windows(
    txt        text,
    pattern    text,
    windows    int4[],
    allow_empty boolean DEFAULT false
) RETURNS int4
AS 'MODULE_PATHNAME', 'regex_verify_windows'
LANGUAGE C
STRICT
STABLE
PARALLEL SAFE;
