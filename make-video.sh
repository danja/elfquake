#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./make-video.sh [input_dir] [output_mp4] [fps]

Defaults:
  input_dir  data/derived/sim/mountain_256x256_seed42_10000.heatmaps
  output_mp4 data/derived/sim/mountain_256x256_seed42_10000.mp4
  fps        20

Example:
  ./make-video.sh \
    data/derived/sim/mountain_256x256_seed42_10000.heatmaps \
    data/derived/sim/mountain_256x256_seed42_10000.mp4 \
    20
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

width="${WIDTH:-256}"
height="${HEIGHT:-256}"
steps="${STEPS:-10000}"
seed="${SEED:-42}"
prefix="data/derived/sim/mountain_${width}x${height}_seed${seed}_${steps}"
input_dir="${1:-${prefix}.heatmaps}"
output_mp4="${2:-${prefix}.mp4}"
fps="${3:-20}"
pattern="sandpile_step_*.png"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg is not installed or not on PATH" >&2
  exit 2
fi

if [[ ! -d "$input_dir" ]]; then
  echo "error: input directory does not exist: $input_dir" >&2
  exit 2
fi

mapfile -t frames < <(find "$input_dir" -maxdepth 1 -type f -name "$pattern" | sort)
if [[ "${#frames[@]}" -eq 0 ]]; then
  echo "error: no frames found in $input_dir matching $pattern" >&2
  exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

index=0
for frame in "${frames[@]}"; do
  printf -v frame_name "frame_%06d.png" "$index"
  ln -s "$(realpath "$frame")" "$tmp_dir/$frame_name"
  index=$((index + 1))
done

mkdir -p "$(dirname "$output_mp4")"

ffmpeg \
  -y \
  -hide_banner \
  -loglevel warning \
  -framerate "$fps" \
  -i "$tmp_dir/frame_%06d.png" \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  "$output_mp4"

echo "frames: ${#frames[@]}"
echo "fps: $fps"
echo "output: $output_mp4"
