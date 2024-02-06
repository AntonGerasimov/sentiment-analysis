import sys

import requests
import zipfile
import io
import os

import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = "settings.json"

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = os.path.join(ROOT_DIR, conf['general']['raw_data_dir']) 

# Create the save path if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

url_with_train = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_train_dataset.zip"

csv_train_filename = conf['train']['table_name']

# Create the save path if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

response_train = requests.get(url_with_train)

# Check if the request was successful (status code 200)
if response_train.status_code == 200:
    # Unzip the content
    with zipfile.ZipFile(io.BytesIO(response_train.content)) as z:
        # Extract only the CSV file from the first item in the zip file
        csv_file_path = os.path.join(DATA_DIR, csv_train_filename)
        with open(csv_file_path, "wb") as csv_file:
            csv_file.write(z.read(z.namelist()[0]))
        logging.info(f"Download and extraction successful. CSV file saved at: {csv_file_path}")
else:
    logging.info(f"Failed to download file. Status code: {response_train.status_code}")
    

url_with_test = "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_test_dataset.zip"
csv_test_filename = conf['inference']['table_name']

response_test = requests.get(url_with_test)

# Check if the request was successful (status code 200)
if response_test.status_code == 200:
    # Unzip the content
    with zipfile.ZipFile(io.BytesIO(response_test.content)) as z:
        # Extract only the CSV file from the first item in the zip file
        csv_file_path = os.path.join(DATA_DIR, csv_test_filename)
        with open(csv_file_path, "wb") as csv_file:
            csv_file.write(z.read(z.namelist()[0]))
        logging.info(f"Download and extraction successful. CSV file saved at: {csv_file_path}")
else:
    logging.info(f"Failed to download test file. Status code: {response_test.status_code}")