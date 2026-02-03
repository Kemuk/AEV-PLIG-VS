#!/bin/bash
# Download datasets for AEV-PLIG-VS training
#
# Usage examples:
#   # Test all URLs without downloading
#   ./scripts/download_data.sh --spider
#
#   # Test with 4 parallel checks
#   ./scripts/download_data.sh --spider --threads 4
#
#   # Download all datasets with 4 parallel workers + extract
#   ./scripts/download_data.sh --threads 4 --extract
#
#   # Download single dataset
#   ./scripts/download_data.sh --dataset pdbbind --extract
#
#   # Just download without extraction
#   ./scripts/download_data.sh --threads 2
#
#   # Custom output directory
#   ./scripts/download_data.sh --output /mnt/data --threads 4 --extract

set -euo pipefail

# Dataset definitions: URL|FILENAME|EXTRACT_DIR
DATASETS="
https://huggingface.co/datasets/Kingldore/aev-plig/blob/main/PDBbind_v2020_refined.tar.gz|pdbbind_refined.tar.gz|pdbbind/refined-set
https://huggingface.co/datasets/Kingldore/aev-plig/blob/main/PDBbind_v2020_other_PL.tar.gz|pdbbind_general.tar.gz|pdbbind/general-set
http://bindingnet.huanglab.org.cn/api/api/download/binding_database|bindingnet.tar.gz|bindingnet/from_chembl_client
https://www.bindingdb.org/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/data/surflex/surflex.tar|bindingdb.tar|bindingdb/surflex
"

# Defaults
OUTPUT_DIR="data"
THREADS=1
EXTRACT=false
SPIDER=false

show_help() {
  cat << EOF
Usage: ${0##*/} [OPTIONS]

Download datasets for AEV-PLIG-VS training

Datasets:
  - PDBbind (refined + general)
  - BindingNet
  - BindingDB-DCS

Options:
  -d, --dataset DATASET   Select dataset (pdbbind|bindingnet|bindingdb|all) [default: all]
  -t, --threads N         Parallel downloads (default: 1)
  -e, --extract           Extract archives after download
  -s, --spider            Test mode - verify URLs without downloading
  -o, --output DIR        Output directory (default: data/)
  -h, --help              Show this help

Examples:
  # Test all URLs
  ${0##*/} --spider

  # Download with 4 parallel threads + extract
  ${0##*/} --threads 4 --extract

  # Download single dataset
  ${0##*/} --dataset pdbbind --extract
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--dataset) DATASET="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    -e|--extract) EXTRACT=true; shift ;;
    -s|--spider) SPIDER=true; shift ;;
    -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help) show_help; exit 0 ;;
    *) echo "Unknown option: $1" >&2; show_help; exit 1 ;;
  esac
done

# Process function for one dataset (download + extract)
process_dataset() {
  IFS='|' read -r url filename extract_dir <<< "$1"
  
  # Spider mode: check if URL exists without downloading
  if [[ "$SPIDER" == "true" ]]; then
    echo "Checking: $url"
    wget --spider "$url" 2>&1 | grep -E "(Remote file exists|Length:|HTTP)" || echo "  [checking...]"
    return
  fi
  
  # Create output directory
  mkdir -p "$OUTPUT_DIR/$extract_dir"
  
  # Download with resume support
  echo "Downloading: $filename"
  wget -c --show-progress "$url" -O "$OUTPUT_DIR/$filename" 2>&1
  
  # Extract if requested
  if [[ "$EXTRACT" == "true" ]]; then
    echo "Extracting: $filename -> $extract_dir"
    if [[ "$filename" == *.tar.gz ]]; then
      tar -xzf "$OUTPUT_DIR/$filename" -C "$OUTPUT_DIR/$extract_dir"
    elif [[ "$filename" == *.tar ]]; then
      tar -xf "$OUTPUT_DIR/$filename" -C "$OUTPUT_DIR/$extract_dir"
    fi
    echo "Extracted: $filename"
  fi
}

# Export function and variables for xargs subshells
export -f process_dataset
export OUTPUT_DIR EXTRACT SPIDER

# Filter datasets if specific one requested
FILTERED_DATASETS="$DATASETS"
if [[ "${DATASET:-all}" != "all" ]]; then
  FILTERED_DATASETS=$(echo "$DATASETS" | grep -i "$DATASET" || true)
  if [[ -z "$FILTERED_DATASETS" ]]; then
    echo "Error: No datasets match '$DATASET'" >&2
    echo "Available: pdbbind, bindingnet, bindingdb" >&2
    exit 1
  fi
fi

# Execute: pipe datasets through xargs for parallel processing
echo "$FILTERED_DATASETS" | grep -v '^$' | xargs -P "$THREADS" -I {} bash -c 'process_dataset "$@"' _ {}

echo ""
echo "Done!"
