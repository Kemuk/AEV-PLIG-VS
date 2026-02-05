#!/usr/bin/env bash
set -euo pipefail

# =========================
# Dataset definitions
# URL|ARCHIVE|EXTRACT_DIR
# =========================
DATASETS="
https://huggingface.co/datasets/Kingldore/aev-plig/resolve/main/PDBbind_v2020_refined.tar.gz|pdbbind_refined.tar.gz|pdbbind/refined-set
https://huggingface.co/datasets/Kingldore/aev-plig/resolve/main/PDBbind_v2020_other_PL.tar.gz|pdbbind_general.tar.gz|pdbbind/general-set
https://huggingface.co/datasets/Kingldore/aev-plig/resolve/main/bindingnet.tar.gz|bindingnet.tar.gz|bindingnet/from_chembl_client
https://huggingface.co/datasets/Kingldore/aev-plig/resolve/main/bindingdb.tar.gz|bindingdb.tar.gz|bindingdb/surflex
"

# =========================
# Defaults
# =========================
OUTPUT_DIR="data"
THREADS=1
MODE="all"          # all | download | extract
SPIDER=false
DATASET="all"

# =========================
# Helpers
# =========================
archive_exists() {
  [[ -f "$1" ]]
}

dataset_extracted() {
  [[ -f "$1/.extracted" ]]
}

get_download_format() {
  while IFS='|' read -r url filename extract_dir; do
    [[ -z "$url" ]] && continue
    echo "$url"
    echo "  dir=$OUTPUT_DIR"
    echo "  out=$filename"
  done <<< "$FILTERED"
}

download_all() {
  if [[ "$SPIDER" == true ]]; then
    while IFS='|' read -r url filename extract_dir; do
      [[ -z "$url" ]] && continue
      echo "Checking: $url"
      wget --spider "$url" 2>&1 | grep -E "HTTP|Length" || true
    done <<< "$FILTERED"
    return
  fi

  echo "Downloading datasets..."
  get_download_format | aria2c \
    -j "$THREADS" \
    -c \
    -x 16 \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --input-file=-
}

extract_dataset() {
  local archive="$1" outdir="$2"
  local marker="$outdir/.extracted"
  local progress_marker="$outdir/.extracting"

  if dataset_extracted "$outdir"; then
    echo "Already extracted: $outdir"
    return
  fi

  if ! archive_exists "$archive"; then
    echo "Archive not found: $archive" >&2
    return 1
  fi

  mkdir -p "$outdir"
  
  case "$archive" in
    *.tar.gz)
      local size=$(pigz -l "$archive" | tail -1 | awk '{print $2}')
      if [[ -f "$progress_marker" ]]; then
        echo "Resuming: $(basename "$archive") → $outdir"
        pigz -dc -p "$THREADS" "$archive" | pv -s "$size" | tar --skip-old-files -xf - -C "$outdir"
      else
        echo "Extracting: $(basename "$archive") → $outdir"
        touch "$progress_marker"
        pigz -dc -p "$THREADS" "$archive" | pv -s "$size" | tar -xf - -C "$outdir"
      fi
      rm -f "$progress_marker"
      ;;
    *)
      echo "Unsupported archive: $archive" >&2
      return 1
      ;;
  esac

  touch "$marker"
}

extract_all() {
  echo "Extracting datasets..."
  while IFS='|' read -r url filename extract_dir; do
    [[ -z "$url" ]] && continue
    extract_dataset "$OUTPUT_DIR/$filename" "$OUTPUT_DIR/$extract_dir"
  done <<< "$FILTERED"
}

# =========================
# CLI
# =========================
show_help() {
  cat << EOF
Usage: ${0##*/} [OPTIONS]

Options:
  -d, --dataset DATASET   pdbbind | bindingnet | bindingdb | all
  -t, --threads N         Parallel workers (default: 1)
  -o, --output DIR        Output directory (default: data/)
      --download-only     Only download archives
      --extract-only      Only extract existing archives
  -s, --spider            Check URLs only
  -h, --help              Show help
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dataset) DATASET="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    -o|--output)  OUTPUT_DIR="$2"; shift 2 ;;
    --download-only) MODE="download"; shift ;;
    --extract-only)  MODE="extract"; shift ;;
    -s|--spider) SPIDER=true; shift ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# =========================
# Dataset filtering
# =========================
FILTERED="$DATASETS"
if [[ "$DATASET" != "all" ]]; then
  FILTERED=$(echo "$DATASETS" | grep -i "$DATASET" || true)
  [[ -z "$FILTERED" ]] && { echo "Unknown dataset: $DATASET" >&2; exit 1; }
fi

mkdir -p "$OUTPUT_DIR"

# =========================
# Main execution
# =========================
if [[ "$SPIDER" == true ]]; then
  download_all
else
  case "$MODE" in
    download)
      download_all
      ;;
    extract)
      extract_all
      ;;
    all)
      download_all
      extract_all
      ;;
  esac
fi

echo "Done."
