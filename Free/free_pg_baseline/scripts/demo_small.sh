#!/usr/bin/env bash
set -euo pipefail

DSN=${DSN:-"postgresql://localhost/postgres"}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Using DSN=${DSN}"

TMP_CSV="$(mktemp)"
cat > "${TMP_CSV}" <<'EOF'
text
hello world
I-5 NB closed
CA-274 reopened quickly
ego sum
EOF

python -m freepg.cli ingest --dsn "${DSN}" --csv "${TMP_CSV}" --text-col text
python -m freepg.cli build-keys --dsn "${DSN}"
python -m freepg.cli build-postings --dsn "${DSN}"
python -m freepg.cli query --dsn "${DSN}" --regex "\yCA\-274\y" --show-plan

rm -f "${TMP_CSV}"
