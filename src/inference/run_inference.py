import pandas as pd

import sys

import os

import json

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pickle

from datetime import datetime

from sklearn.metrics import accuracy_score, recall_score, f1_score

from typing import List

def identity_tokenizer(text):
    return text

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.path.join(ROOT_DIR, "settings.json")

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, conf['general']['processed_data_dir']) 
MODEL_DIR = os.path.join(ROOT_DIR, conf['general']['models_dir'])
VECTORIZER_DIR = os.path.join(ROOT_DIR, conf['general']['vectorizer_dir'])

PREDICTIONS_DIR = os.path.join(ROOT_DIR, conf['general']['predictions_dir'])
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

test_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, conf['inference']['test_name']))

def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '.pickle') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '.pickle'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)

def get_all_model_paths() -> List[str]:
    """Gets the paths of all saved models"""
    model_paths = []
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            #if filename.endswith('.pkl'):
            model_paths.append(os.path.join(dirpath, filename))
    return model_paths

model_path = get_latest_model_path()

with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

vectorizer_path = os.path.join(VECTORIZER_DIR, "vectorizer.pkl")
    
with open(vectorizer_path, 'rb') as file:
    loaded_vectorizer = pickle.load(file)    

X_test = loaded_vectorizer.transform(test_data['tokens']) 

test_data['prediction'] = loaded_model.predict(X_test)

predictions = test_data['prediction']

ground_truth = test_data['sentiment']


def generate_metric_text(model_name, metric_name, metric_value):
    logging.info(f'{metric_name}={metric_value}. Saving it to outputs txt file')
    return f"The {metric_name} of model {model_name} is {metric_value} \n"

metric_path =  os.path.join(PREDICTIONS_DIR, conf['inference']['metrics_file'])

def save_metrics(ground_truth, prediction):
    metric_text = ""
    model_name = "Logistic regression"
    accuracy = round(accuracy_score(ground_truth, prediction)*100,1)
    metric_text += generate_metric_text(model_name, "accuracy", accuracy)
    
    pos_label = 'negative'
    recall = round(recall_score(ground_truth, prediction, pos_label=pos_label)*100, 1)
    metric_text += generate_metric_text(model_name, "recall", recall)
    f1 = round(f1_score(ground_truth, prediction, pos_label=pos_label)* 100, 1)
    metric_text += generate_metric_text(model_name, "f1 score", f1)
    
    with open(metric_path, 'w') as file:
        file.write(metric_text)

save_metrics(ground_truth, predictions)

csv_path = os.path.join(PREDICTIONS_DIR, conf['inference']['predictions_file'])
csv_content = test_data.to_csv(index=False)

with open(csv_path, 'w') as file:
    file.write(csv_content)