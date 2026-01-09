import os

# Default parameters for key discovery and postings build
DEFAULT_C = 0.1
DEFAULT_LMAX = 10
DEFAULT_SAMPLE_ROWS = 3_000_000
DEFAULT_USE_SHELL = True
DEFAULT_SUBSTRING_CHOICES = 2  # number of substrings to pick when substituting
DEFAULT_LITERAL_MIN_LEN = 2
DEFAULT_INDEX_NOTES = ""

# Spill-to-disk behaviour for key discovery counting
DEFAULT_SHARDS = 16
DEFAULT_SPILL_THRESHOLD = int(os.environ.get("FREEPG_SPILL_THRESHOLD", "200000"))
DEFAULT_POSTINGS_SHARDS = int(os.environ.get("FREEPG_POSTINGS_SHARDS", "128"))

# Batch sizes
FETCH_BATCH = 10_000
INSERT_BATCH = 5_000

# Query materialization thresholds
ARRAY_ANY_MAX = 200_000
POSTINGS_CACHE_SIZE = 50_000  # legacy, not used for byte budgeting
CAND_RATIO_GATE = float(os.environ.get("FREEPG_CAND_RATIO_GATE", "0.5"))
POSTINGS_CACHE_MAX_BYTES = int(os.environ.get("FREEPG_CACHE_MAX_BYTES", str(1 * 1024 * 1024 * 1024)))  # 1GB default
POSTINGS_CACHE_MAX_ENTRY_BYTES = int(os.environ.get("FREEPG_CACHE_MAX_ENTRY_BYTES", str(32 * 1024 * 1024)))  # 32MB
LEAF_TEMP_THRESHOLD = int(os.environ.get("FREEPG_LEAF_TEMP_THRESHOLD", "5000"))

# Aho–Corasick chunking
AHO_CHUNK_SIZE = 50_000
DEFAULT_BACKEND = "auto"  # enum_ngrams, ac_chunk, bucket
DEFAULT_TMPDIR = os.environ.get("FREEPG_TMPDIR", "")
DEFAULT_LOGGED = False
DEFAULT_SORT_CMD = os.environ.get("FREEPG_SORT_CMD", "sort")
DEFAULT_SORT_MEM = os.environ.get("FREEPG_SORT_MEM", "")

# Misc
LOG_PREFIX = "[freepg]"


def env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def make_index_id(c: float, lmax: int, use_shell: bool, discovery_rows: int, docs_table: str = "docs") -> str:
    shell = "shell" if use_shell else "min"
    return f"{docs_table}_c{c}_l{lmax}_{shell}_disc{discovery_rows}"
