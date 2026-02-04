#!/usr/bin/env bash
set -euo pipefail

# =========================
# Dataset definitions
# URL|ARCHIVE|EXTRACT_DIR
# =========================
DATASETS="
https://huggingface.co/datasets/Kingldore/aev-plig/resolve/main/PDBbind_v2020_refined.tar.gz|pdbbind_refined.tar.gz|pdbbind/refined-set
https://huggingface.co/datasets/Kingldore/aev-plig/resolve/main/PDBbind_v2020_other_PL.tar.gz|pdbbind_general.tar.gz|pdbbind/general-set
https://huggingface.co/datasets/Kingldore/aev-plig/blob/main/bindingnet.tar.gz|bindingnet.tar.gz/from_chembl_client
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

download_dataset() {
  local url="$1" archive="$2"

  if archive_exists "$archive"; then
    echo "Archive exists: $(basename "$archive")"
    return
  fi

  echo "Downloading: $(basename "$archive")"
  wget -c --show-progress "$url" -O "$archive"
}

extract_dataset() {
  local archive="$1" outdir="$2"
  local marker="$outdir/.extracted"

  if dataset_extracted "$outdir"; then
    echo "Already extracted: $outdir"
    return
  fi

  mkdir -p "$outdir"
  echo "Extracting: $(basename "$archive") → $outdir"

  case "$archive" in
    *.tar.gz) tar -xzf "$archive" -C "$outdir" ;;
    *.tar)    tar -xf  "$archive" -C "$outdir" ;;
    *) echo "Unsupported archive: $archive" >&2; exit 1 ;;
  esac

  touch "$marker"
}

process_dataset() {
  IFS='|' read -r url filename extract_dir <<< "$1"

  local archive="$OUTPUT_DIR/$filename"
  local target="$OUTPUT_DIR/$extract_dir"

  if [[ "$SPIDER" == true ]]; then
    echo "Checking: $url"
    wget --spider "$url"
    return
  fi

  case "$MODE" in
    download)
      download_dataset "$url" "$archive"
      ;;
    extract)
      extract_dataset "$archive" "$target"
      ;;
    all)
      download_dataset "$url" "$archive"
      extract_dataset "$archive" "$target"
      ;;
  esac
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

export -f process_dataset download_dataset extract_dataset \
         archive_exists dataset_extracted
export OUTPUT_DIR MODE SPIDER

mkdir -p "$OUTPUT_DIR"

echo "$FILTERED" | grep -v '^$' \
  | xargs -P "$THREADS" -I {} bash -c 'process_dataset "$@"' _ {}

echo "Done."
