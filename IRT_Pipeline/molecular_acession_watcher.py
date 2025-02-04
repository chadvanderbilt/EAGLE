#%%
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)
import pandas as pd
import os
import re
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from decouple import config
from django.core import management
import logging
import requests
from urllib.parse import urlencode
from urllib.parse import quote
import json
import subprocess
import argparse
from api_call import get_data_copath_api
from get_token import get_token
#%%
def main(input_csv, output_csv, prefix, start_accession=None):
    print('imported')
    token = get_token()

    df_towrite = pd.read_csv(input_csv)

    # Determine the starting accession number
    if start_accession != 1:
        if len(df_towrite) >= 1:
            last_row_value = df_towrite['Mnumber'].iloc[-1]  # Get the last row value
            start_accession = int(last_row_value.split('-')[1])
        else:
            start_accession = 1

    print(f'Starting from {prefix}{start_accession}')
    time.sleep(5)  # Pause for 5 seconds

    exception_counter = 0
    while True:
        try:
            case_id = f"{prefix}{start_accession}"
            print(case_id)

            df_check = pd.read_csv(input_csv)

            if case_id in df_check['Mnumber'].values:
                start_accession += 1
                exception_counter = 0
                continue  # Skip to the next item

            print(case_id)
            url_new = f"https://pathapi.aws.mskcc.org/api/accession/{case_id}/diagnosis"
            headers_new = {'Authorization': f'Bearer {token}'}

            response_new = requests.get(url_new, headers=headers_new, verify=False)
            response_new_json = response_new.json()

            df = pd.json_normalize(response_new_json['parts'])
            df_test = pd.json_normalize(response_new_json['retrieval_flags'])

            df = df[['datetime_rec', 'part_description', 'parttype.name']]
            df['category_description'] = df_test['category.description']

            pattern = r'([SCFH]\d{2}-\d+)'
            df['Snumber'] = df['part_description'].str.extract(pattern)

            pattern_block_id = r'[SCFH]\d{2}-\d+/([^,]*)'
            df['block_id'] = df['part_description'].str.extract(pattern_block_id)
            df['Mnumber'] = case_id

            df_towrite = df_towrite.append(df, ignore_index=True)
            df_towrite.to_csv(output_csv, index=False)

            start_accession += 1
            exception_counter = 0
            print(df.head())
            print(start_accession)
            # time.sleep(1)
        except Exception as e:
            start_accession += 1
            exception_counter += 1
            print(f"An error occurred: {str(e)}. Exception number: {exception_counter}")
            if exception_counter >= 7:
                print("7 exceptions in a row occurred. Exiting the function.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run molecular slide info script")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file')
    parser.add_argument('--prefix', type=str, required=True, help='Prefix for the case IDs')
    parser.add_argument('--start_accession', type=int, help='Optional starting accession number')

    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.prefix, args.start_accession)
