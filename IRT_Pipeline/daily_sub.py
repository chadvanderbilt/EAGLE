#%%
# This script runs hourly to monitor and identify newly scanned slides, extracting relevant metadata
# for downstream processing in the IRT pipeline. It collects slide information, integrates data from APIs,
# and generates a manifest file to track processed slides.
#
# **Assumptions:**
# - This script assumes the presence of a standardized directory structure for scanned slides.
# - Adjustments may be required to fit specific institutional environments.
#
# **Script Overview:**
# - Monitors specified directories for newly scanned slides.
# - Fetches additional slide metadata using API calls.
# - Updates a manifest file with relevant fields for downstream IRT processing.
# - Filters out already processed slides to avoid duplication.
#
# **Libraries Used:**
# - api_call, get_token: Custom modules for API data retrieval and authentication.
#
# **Arguments:**
# - Date: Date of the slides to process, formatted as YYYY-MM-DD.
# - --output_dir: Directory where the manifests will be saved.
#
# **Usage Example:**
# python daily_sub.py 2024-04-25 --output_dir /path/to/save/manifests
#%%

import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from decouple import config
from django.core import management
import logging
import requests
from urllib.parse import urlencode, quote
import json
import pandas as pd
import subprocess
import argparse
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from api_call import get_data_copath_api
from get_token import get_token

print('imported')

# Create the parser
parser = argparse.ArgumentParser(description='Process a date to identify newly scanned slides.')

# Add the arguments
parser.add_argument('--date', metavar='date', type=str, help='The date to process (format: YYYY-MM-DD)')
parser.add_argument('--output_dir', type=str, required=True, help='Directory where the manifests will be saved')

# Parse the arguments
args = parser.parse_args()

# Access the date and output directory passed as command-line arguments
Date = args.Date
output_dir = args.output_dir

# Get authorization token
token = get_token()

# Create the directory path for storing the manifest
dir_path = os.path.join(output_dir, Date)
os.makedirs(dir_path, exist_ok=True)
output_file_path = os.path.join(dir_path, f"manifest_{Date}.csv")

# Identify newly scanned slides using a bash command
# Assumes the directory structure is standardized and the date is part of the directory name
# Modify the prefixes and paths as per your institutional setup
bash_command = f"""
for prefix in SS NPTHE IPTH; do
    for dir in /pesgisipth/${{prefix}}*; do
        if [ -d "${{dir}}/{Date}" ]; then
            ls "${{dir}}/{Date}"
        fi
    done 
done 
"""

# Execute the command and capture the output
output = subprocess.check_output(bash_command, shell=True).decode('utf-8')

# Split the output into lines and create a DataFrame
output_lines = output.split('\n')
df = pd.DataFrame(output_lines, columns=['slide_file_name'])

# Initialize columns for slide metadata
df['service_name'] = ""
df['sub_specialty'] = ""
df['case_accessionDate'] = ""
df['case_id_slide'] = ""
df['part_id'] = ""
df['block_name'] = ""
df['part_description'] = ""
df['stain'] = ""
df['slide_barcode'] = ""
df['slide_id'] = ""
df['scanner_id'] = ""
df['Molecular_Block'] = ""
df['Molecular_LUAD'] = ""

# Remove empty rows
df = df[df['slide_file_name'] != ""]

# Check for existing manifest to avoid reprocessing
if os.path.exists(output_file_path):
    df_current = pd.read_csv(output_file_path)
    df_current = df_current[pd.notna(df_current['service_name'])]
    df_current = df_current[df_current['service_name'] != "unknown"]
    if 'Molecular_LUAD' not in df_current.columns:
        df_current['Molecular_LUAD'] = ''
    df = df[~df['slide_file_name'].isin(df_current['slide_file_name'])]
else:
    df_current = pd.DataFrame()

# Concatenate new and existing dataframes
df = pd.concat([df_current, df])

# Iterate through slides to fetch metadata from API
for index, row in df.iterrows():
    if row['service_name'] != "":
        continue
    print(f"Processing slide {index + 1}/{df.shape[0]}: {row['slide_file_name']}")
    try:
        final = get_data_copath_api(row['slide_file_name'], token)

        # Populate DataFrame with API response
        df.at[index, 'service_name'] = final[0]
        df.at[index, 'sub_specialty'] = final[1]
        df.at[index, 'block_name'] = ''.join(final[4])
        df.at[index, 'case_accessionDate'] = ''.join(final[5])
        df.at[index, 'case_id_slide'] = ''.join(final[6])
        df.at[index, 'part_description'] = ''.join(final[7])
        df.at[index, 'stain'] = ''.join(final[8])
        df.at[index, 'slide_barcode'] = ''.join(final[9])
        df.at[index, 'slide_id'] = ''.join(final[10])
        df.at[index, 'scanner_id'] = ''.join(final[11])
        df.at[index, 'part_id'] = ''.join(final[12])
        df.at[index, 'Molecular_Block'] = final[13]
        df.at[index, 'Molecular_LUAD'] = final[14]

        # Save updated manifest to CSV
        df.to_csv(output_file_path, index=False)
    except Exception as e:
        print(f"An error occurred while processing {row['slide_file_name']}: {e}")
        continue