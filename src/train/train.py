import numpy as np
import pandas as pd

import sys

import os

import json

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

import pickle

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

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VECTORIZER_DIR, exist_ok=True)

logging.info('Downloading train data')
train_data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, conf['train']['train_name']))

max_features = int(conf['tf_idf']['max_features'])
min_df = int(conf['tf_idf']['min_df'])
max_df = float(conf['tf_idf']['max_df'])

logging.info('Starting vectorization')
tfidf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, max_features=max_features, lowercase=False)

tfidf_vectorizer = TfidfVectorizer (max_features=2000, min_df=min_df, max_df=max_df)

tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_data['tokens'])
X_train_tfidf_vector =  tfidf_matrix_train.toarray()
logging.info('Vectorization completed')

y_train = train_data['sentiment']

max_iter = int(conf['log_reg']['max_iter'])
C = float(conf['log_reg']['C'])
penalty = conf['log_reg']['penalty']

logging.info('Training the model')
model_lg = LogisticRegression(max_iter = max_iter, C = C, penalty = penalty, solver = 'liblinear', random_state = 42)
model_lg.fit(X_train_tfidf_vector, y_train)

model_lg_acc = cross_val_score(estimator=model_lg, X=X_train_tfidf_vector, y=y_train, cv=5, n_jobs=-1)
mean_acc = np.mean(model_lg_acc)
logging.info(f"The model mean cross validation accuracy is {round(mean_acc*100,1)}")

model = model_lg
model_filename = os.path.join(MODEL_DIR, conf['general']['model_name'])
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

vectorizer = tfidf_vectorizer
vectorizer_filename = os.path.join(VECTORIZER_DIR, conf['general']['vectorizer_name'])
with open(vectorizer_filename, 'wb') as file:
    pickle.dump(vectorizer, file)