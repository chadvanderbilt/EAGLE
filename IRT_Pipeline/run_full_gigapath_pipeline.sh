#!/bin/bash

# ===============================================================
# EGFR Slide Processing Pipeline
# ===============================================================
# This script runs a full pipeline to process histopathology slides
# and extract EGFR mutation results. The steps include:
# 
# 1. **Molecular Accession Watching:**
#    - Tracks molecular accessions and updates input CSV files.
# 
# 2. **Filtering for EGFR Tests:**
#    - Filters the molecular data for slides tested via "EGFR via Idylla".
# 
# 3. **Slide Processing with Gigapath:**
#    - Processes each slide using the Gigapath model to extract relevant results.
# 
# Required Arguments:
#   --base_dir       : Base directory where data will be stored.
#   --python_path    : Path to the Python executable.
#   --rscript_path   : Path to the Rscript executable.
#   --repo_path      : Path to the cloned Git repository containing the scripts.
# 
# Example Usage:
# ./run_pipeline.sh --base_dir /path/to/base --python_path /usr/bin/python3 \
#                   --rscript_path /usr/bin/Rscript --repo_path /path/to/repo
# ===============================================================
# In practice is run as a cron job hourly to run every hour or can be 
# run manually as needed. The Results will then be generated automatically,
# which then can be used for further analysis. 
# Checkpoints are also saved in the checkpoints folder which can be downloaded 
# our associated huggingface repository.
# ===============================================================
# Exit immediately if a command exits with a non-zero status
set -e

# Function to print usage
usage() {
    echo "Usage: $0 --base_dir <BASE_DIR> --python_path <PYTHON_PATH> --rscript_path <RSCRIPT_PATH> --repo_path <REPO_PATH>"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --base_dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --python_path)
            PYTHON_PATH="$2"
            shift 2
            ;;
        --rscript_path)
            RSCRIPT_PATH="$2"
            shift 2
            ;;
        --repo_path)
            REPO_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Ensure required arguments are provided
if [[ -z "$BASE_DIR" || -z "$PYTHON_PATH" || -z "$RSCRIPT_PATH" || -z "$REPO_PATH" ]]; then
    echo "Missing required arguments."
    usage
fi

# Create necessary directories if they don't exist
mkdir -p "$BASE_DIR/molecular_watcher"
mkdir -p "$BASE_DIR/slides_to_run"
mkdir -p "$BASE_DIR/EGFR_results"
mkdir -p "$BASE_DIR/tmp_files"
mkdir -p "$BASE_DIR/checkpoints"

# Define CSV file paths
MOLECULAR_WATCHER_CSV="$BASE_DIR/molecular_watcher/Mnumber_Snumber_block_2025.csv"
EGFR_RESULTS_CSV="$BASE_DIR/EGFR_results/gigapath_results.csv"
SLIDES_TO_PROCESS_CSV="$BASE_DIR/slides_to_run/slides_to_processes.csv"
MOLECULAR_TEST_NAME="EGFR via Idylla"

# Run molecular accession watcher
# This assumes a specific ACCESSION_NUMBER format starting with "M25-" and incrementing by 1
"$PYTHON_PATH" "$REPO_PATH/molecular_acession_watcher.py" \
    --input_csv "$MOLECULAR_WATCHER_CSV" \
    --output_csv "$MOLECULAR_WATCHER_CSV" \
    --prefix M25- \
    --start_accession 1

# Run EGFR filtering script
# This script filters the molecular data for slides tested via "EGFR via Idylla" which is our Internal Test Name. Adjust as needed.
"$RSCRIPT_PATH" "$REPO_PATH/filter_molecular_for_EGFR.R" \
    --input_csv "$MOLECULAR_WATCHER_CSV" \
    --directory_out "$BASE_DIR/slides_to_run" \
    --slide_data_dir <PATH_TO_SLIDE_DAILY_MANIFESTS> \
    --test_to_filter $MOLECULAR_TEST_NAME \
    --completed "$EGFR_RESULTS_CSV"

# Read the header of slides_to_process.csv
HEADER=$(head -n 1 "$SLIDES_TO_PROCESS_CSV")

# Process each slide

tail -n +2 "$SLIDES_TO_PROCESS_CSV" | while IFS= read -r line; do
    TEMP_CSV="$BASE_DIR/tmp_files/temp_slide.csv"
    echo "$HEADER" > "$TEMP_CSV"
    echo "$line" >> "$TEMP_CSV"

    # Run Gigapath processing
    "$PYTHON_PATH" "$REPO_PATH/run_gigapath_args.py" \
        --batch_size 15 \
        --outdir "$BASE_DIR/EGFR_results" \
        --outname gigapath_results.csv \
        --workers 10 \
        --tile_checkpoint "$BASE_DIR/checkpoints/gigapath_ft_checkpoint_tile_020.pth" \
        --slide_checkpoint "$BASE_DIR/checkpoints/gigapath_ft_checkpoint_slide_020.pth" \
        --test_csv "$TEMP_CSV" \
        --gpu 2 \
        --tmp_file_holding "$BASE_DIR/tmp_files"

    # Remove temporary file after processing
    rm "$TEMP_CSV"
done

echo "Pipeline execution completed successfully!"
