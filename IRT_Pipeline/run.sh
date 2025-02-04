#!/bin/bash

# This script is designed to be run as a cron job to monitor newly scanned slides
# and generate manifests for downstream processing in the IRT pipeline.
#
# **Functionality:**
# - Stops any running instance of `daily_sub.py` to avoid conflicts.
# - Executes the Python monitoring script (`daily_sub.py`) to collect slide data.
# - Logs outputs to a timestamped log file.
#
# **Usage:**
# - This script should be scheduled to run every hour, offset by 30 minutes from the `run_full_gigapath_pipeline.sh` cron job.
# - Example cron entry: `30 * * * * /path/to/this_script.sh`

# Set the base path for logs and outputs
BASE_PATH="/home/chad/production"
LOG_DIR="$BASE_PATH/logs/irt_monitor"
SCRIPT_DIR="$BASE_PATH/scripts"
OUTPUT_DIR="$BASE_PATH/slide_data"

# Create necessary directories if they don't exist
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR
mkdir -p $SCRIPT_DIR

# Assign the current date in the format YYYY-MM-DD to the Date variable
Date=$(date +%Y-%m-%d)
Date_time=$(date +%Y-%m-%d_%H-%M-%S)

# Stop any currently running instance of daily_sub.py
pkill -f "daily_sub.py"

# Pause to ensure the process is fully terminated
sleep 10

# Full path to the Python interpreter in the osteo environment
PYTHON="/path/to/python"

# Navigate to the script directory
cd $SCRIPT_DIR

# Run the Python script with the date and output directory arguments, logging the output
nohup $PYTHON daily_sub.py $Date --output_dir $OUTPUT_DIR > $LOG_DIR/$Date_time.log &

# nohup filter_lung.R --date --date $Date > /home/chad/production/outputs/filter_lung_$Date_time.txt &
# nohup python /production_vitbase_gm_with_args.py --today $Date > $LOG_DIR/production_vitbase_gm_with_args_$Date_time.txt &