#%%
# This script fetches diagnostic data for molecular cases from an API, processes the data,
# and appends it to an output CSV file. The script runs in a loop, automatically handling errors
# and continuing from where it left off if interrupted.
# This assumes that there is an association between the molecular accession number and the 
# surgical pathology accession number available through the API.  Please adjust to the formatting
# used in your institution. 

# **Functions Overview:**
# - main: Retrieves diagnostic information using case IDs, processes the API response,
#         and writes results to a CSV file.
#
# **Non-Standard Library Descriptions:**
# - api_call: Custom module, likely contains the `get_data_copath_api` function for additional API calls.
# - get_token: Custom module used to obtain an authentication token for API access.
#
# **Configuration Notes:**
# - Update the API URL (`url_new`) with your API endpoint.
# - Ensure the `get_token` function is configured correctly to retrieve a valid token.
# - Replace any static values with your environment-specific configurations.
# - SSL verification is disabled for development; enable it in production environments.
#%%

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
import pandas as pd
import os
import re
import time
from watchdog.observers import Observer  # Imported but not used in this script
from watchdog.events import FileSystemEventHandler  # Imported but not used in this script
from decouple import config  # For managing environment variables
import logging
from urllib.parse import urlencode, quote
import json
import subprocess
import argparse
from api_call import get_data_copath_api
from get_token import get_token

def main(input_csv, output_csv, prefix, start_accession=None):
    print('Starting data retrieval...')
    token = get_token()  # Obtain API token

    df_towrite = pd.read_csv(input_csv)

    # Determine the starting accession number
    if start_accession != 1:
        if len(df_towrite) >= 1:
            last_row_value = df_towrite['Mnumber'].iloc[-1]  # Get the last accession number
            start_accession = int(last_row_value.split('-')[1])
        else:
            start_accession = 1

    print(f'Starting from {prefix}{start_accession}')
    time.sleep(5)  # Pause for 5 seconds

    exception_counter = 0
    while True:
        try:
            case_id = f"{prefix}{start_accession}"
            print(f"Processing case ID: {case_id}")

            # Check if the case ID already exists in the CSV
            df_check = pd.read_csv(input_csv)
            if case_id in df_check['Mnumber'].values:
                start_accession += 1
                exception_counter = 0
                continue  # Skip if already processed

            api_url = f"https://your-api-endpoint.com/api/accession/{case_id}/diagnosis"  # Replace with your API URL
            headers = {'Authorization': f'Bearer {token}'}

            response = requests.get(api_url, headers=headers, verify=False)
            response.raise_for_status()  # Raise an error for bad status codes
            response_json = response.json()

            # Process API response data
            df = pd.json_normalize(response_json.get('parts', []))
            df_test = pd.json_normalize(response_json.get('retrieval_flags', []))

            df = df[['datetime_rec', 'part_description', 'parttype.name']]
            df['category_description'] = df_test.get('category.description', None)

            # Extract S-number and block ID using regex
            df['Snumber'] = df['part_description'].str.extract(r'([SCFH]\d{2}-\d+)')
            df['block_id'] = df['part_description'].str.extract(r'[SCFH]\d{2}-\d+/([^,]*)')
            df['Mnumber'] = case_id

            # Append to the output DataFrame and save to CSV
            df_towrite = pd.concat([df_towrite, df], ignore_index=True)
            df_towrite.to_csv(output_csv, index=False)

            print(df.head())
            start_accession += 1
            exception_counter = 0
        
        except Exception as e:
            print(f"An error occurred: {str(e)}. Exception number: {exception_counter + 1}")
            start_accession += 1
            exception_counter += 1
            if exception_counter >= 7:
                print("7 consecutive exceptions occurred. Exiting the script.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch molecular diagnosis data from API")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for the case IDs')
    parser.add_argument('--start_accession', type=int, help='Optional starting accession number')

    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.prefix, args.start_accession)
#%%
