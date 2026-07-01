#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
target_dir="$root_dir/data/derived"
days="7"
apply="false"
include_manifests="false"
list="false"

usage() {
  cat <<'USAGE'
Usage: ./tidy.sh [--apply] [--days N] [--all] [--sim-only] [--include-manifests] [--list]

Delete generated files under data/derived. Raw captures under data/raw are never touched.

Defaults:
  dry run only
  files older than 7 days
  preserve files under data/derived/**/manifests/

Options:
  --apply              Actually delete matching files.
  --days N            Delete files older than N days.
  --all               Delete matching derived files regardless of age.
  --sim-only          Restrict cleanup to data/derived/sim.
  --include-manifests Also delete derived manifest files.
  --list              Print each matching path.
  -h, --help          Show this help.
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --apply)
      apply="true"
      ;;
    --days)
      if [[ "$#" -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
        echo "error: --days requires a non-negative integer" >&2
        exit 2
      fi
      days="$2"
      shift
      ;;
    --all)
      days="all"
      ;;
    --sim-only)
      target_dir="$root_dir/data/derived/sim"
      ;;
    --include-manifests)
      include_manifests="true"
      ;;
    --list)
      list="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! -d "$target_dir" ]]; then
  echo "nothing to tidy: $target_dir does not exist"
  exit 0
fi

find_args=(
  "$target_dir"
  -type f
  \( -name '*.csv' -o -name '*.json' -o -name '*.npy' -o -name '*.png' -o -name '*.mp4' -o -name '*.txt' \)
)

if [[ "$days" != "all" ]]; then
  find_args+=(-mtime +"$days")
fi

if [[ "$include_manifests" != "true" ]]; then
  find_args+=(! -path '*/manifests/*')
fi

count="0"
bytes="0"

while IFS= read -r -d '' path; do
  count="$((count + 1))"
  if size="$(wc -c < "$path" 2>/dev/null)"; then
    bytes="$((bytes + size))"
  fi
  if [[ "$apply" == "true" ]]; then
    rm -f -- "$path"
    if [[ "$list" == "true" ]]; then
      echo "deleted: ${path#$root_dir/}"
    fi
  else
    if [[ "$list" == "true" ]]; then
      echo "would delete: ${path#$root_dir/}"
    fi
  fi
done < <(find "${find_args[@]}" -print0)

if [[ "$apply" == "true" ]]; then
  find "$target_dir" -mindepth 1 -type d -empty -delete
  echo "deleted $count files, approx $bytes bytes"
else
  echo "dry run: $count files, approx $bytes bytes"
  echo "rerun with --apply to delete"
fi
