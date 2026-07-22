#!/usr/bin/env bash
set -euo pipefail

# Download one ISEE CDF unchanged and write a minimal capture record beside it.
: "${URL:?Set URL to one exact ISEE VLF CDF file URL}"
OUTPUT="${OUTPUT:-data/raw/vlf/japan/$(basename "${URL%%\?*}")}"
META="${META:-${OUTPUT}.capture.json}"

mkdir -p "$(dirname "$OUTPUT")"
curl --fail --location --retry 3 --output "$OUTPUT" "$URL"
test -s "$OUTPUT"

python3 - "$URL" "$OUTPUT" "$META" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

url, output, metadata = sys.argv[1:]
path = Path(output)
record = {
    "schema": "elfquake.vlf_cdf_capture.v1",
    "url": url,
    "path": str(path),
    "captured_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "bytes": path.stat().st_size,
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "scientific_use": True,
    "note": "Retain the source archive caveats and station metadata with derived products.",
}
Path(metadata).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, sort_keys=True))
PY
