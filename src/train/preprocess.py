import pandas as pd

import sys

import os

import re

import json


import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = "settings.json"

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

RAW_DATA_DIR = os.path.join(ROOT_DIR, conf['general']['raw_data_dir']) 

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, conf['general']['processed_data_dir']) 

train_data = pd.read_csv(os.path.join(RAW_DATA_DIR, conf['raw']['train_name']))
test_data = pd.read_csv(os.path.join(RAW_DATA_DIR, conf['raw']['test_name']))

common_words_file_path = os.path.join(PROCESSED_DATA_DIR, conf['train']['common_words_file_name'])

with open(common_words_file_path, 'r') as file:
    common_words_content = file.read()

common_words = set(word for word in common_words_content.split('\n') if word)


class preprocess_with_tokens():
    
    def __init__(self, data):
        self.data = data
        self.common_words = common_words
        
    def do_preprocess(self):
        self.preprocess_data()
        return self.data
            
    def preprocess_data(self):
        data = self.data
        data['review'] = data['review'].apply(self.remove_url)
        data['review'] = data['review'].apply(self.remove_html_tags)
        data['review'] = data['review'].apply(self.remove_non_alphanumeric)
        data['review'] = data['review'].apply(self.convert_to_lowercase)
        data['tokens'] = data['review'].apply(self.tokenization)
        data['tokens'] = data['tokens'].apply(self.remove_short_words)
        data['tokens'] = data['tokens'].apply(self.remove_stopwords)
        data['tokens'] = data['tokens'].apply(lambda tokens: self.remove_common_words_from_tokens(tokens, self.common_words))
        data['tokens'] = data['tokens'].apply(self.lemmatize_tokens)
        self.data = data
        
    def remove_url(self, review_text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return re.sub(url_pattern, '', review_text)
    
    def remove_html_tags(self, review_text):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', review_text)
    
    def remove_non_alphanumeric(self, review_text):
        return re.sub(r'[^a-zA-Z]', ' ', review_text)
    
    def convert_to_lowercase(self, review_text):
        review_text = review_text.lower()
        return review_text

    def tokenization(self, review_text):
        return word_tokenize(review_text)
    
    def remove_short_words(self, tokens, min_length=3):
        result = [word for word in tokens if len(word) > min_length]
        return result
    
    def remove_stopwords(self, tokens):
        STOPWORDS = set(stopwords.words('english'))
        result = [i for i in tokens if not i in STOPWORDS]
        return result
    
    def lemmatize_tokens(self, tokens):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in tokens]

    def remove_common_words_from_tokens(self, tokens, common_words):
        return [token for token in tokens if token.lower() not in common_words]



preprocessing_train = preprocess_with_tokens(train_data)
preprocessed_train= preprocessing_train.do_preprocess()
df_train = pd.DataFrame(preprocessed_train)

train_file_name = os.path.join(PROCESSED_DATA_DIR, conf['train']['train_name'])
df_train.to_csv(train_file_name, index=False)

preprocessing_test = preprocess_with_tokens(test_data)
preprocessed_test= preprocessing_test.do_preprocess()
df_test = pd.DataFrame(preprocessed_test)

test_file_name = os.path.join(PROCESSED_DATA_DIR, conf['train']['test_name'])
df_test.to_csv(test_file_name, index=False)