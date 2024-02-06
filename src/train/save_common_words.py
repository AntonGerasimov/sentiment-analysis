import pandas as pd

import sys

import os

import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import json
import logging

def find_common_words(data):
  positive_review = data[data.sentiment == 'positive']['review']
  negative_review = data[data.sentiment == 'negative']['review']
  splited_review = [positive_review, negative_review]
  all_text_sets = [' '.join(examples) for examples in splited_review]

  top_words_sets = []

  for item, text_set in enumerate(all_text_sets):
    top_words_series = pd.Series(text_set.split()).value_counts().head(13)
    top_words = top_words_series.index.tolist()
    top_word_counts = top_words_series.values.tolist()
    top_words_sets.append(set(top_words))

  return set.intersection(*top_words_sets)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.path.join(ROOT_DIR, "settings.json")

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

RAW_DATA_DIR = os.path.join(ROOT_DIR, conf['general']['raw_data_dir']) 

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, conf['general']['processed_data_dir']) 
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

train_file_name = "train.csv"

train_data = pd.read_csv(os.path.join(RAW_DATA_DIR, train_file_name))

class preprocess_data_without_tokens():
    def __init__(self):
        self.data = train_data
        
    def find_data(self):
        self.preprocess_data()
        return self.data
    
    def preprocess_data(self):
        review = self.data['review']
        review = review.apply(self.remove_url)
        review = review.apply(self.remove_html_tags)
        review = review.apply(self.remove_non_alphanumeric) 
        review = review.apply(self.convert_to_lowercase)   
        review = review.apply(self.remove_short_words)   
        review = review.apply(self.remove_stopwords)   
        self.data['review'] = review
        
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
    
    def remove_short_words(self, review_text, min_length=3):
        return ' '.join(word for word in review_text.split() if len(word) > min_length)
    
    def remove_stopwords(self, review_text):
        STOPWORDS = set(stopwords.words('english'))
        words = review_text.split()
        filtered_words = [word for word in words if word.lower() not in STOPWORDS]
        return ' '.join(filtered_words)
    
    def find_common_words(self, data):
        positive_review = data[data.sentiment == 'positive']['review']
        negative_review = data[data.sentiment == 'negative']['review']
        splited_review = [positive_review, negative_review]
        all_text_sets = [' '.join(examples) for examples in splited_review]

        top_words_sets = []

        for item, text_set in enumerate(all_text_sets):
            top_words_series = pd.Series(text_set.split()).value_counts().head(13)
            top_words = top_words_series.index.tolist()
            top_word_counts = top_words_series.values.tolist()
            top_words_sets.append(set(top_words))

        return set.intersection(*top_words_sets)
    
    
     
preprocessing = preprocess_data_without_tokens()

train_data = preprocessing.find_data()
common_words = find_common_words(train_data)
logging.info(f"Found frequently used common words in train data (for negative and positive sentiment): {common_words}")

common_words_file_path = os.path.join(PROCESSED_DATA_DIR, conf['train']['common_words_file_name'])

# Save common_words to the file
with open(common_words_file_path, 'w') as file:
    for i, word in enumerate(common_words):
        file.write(word)
        if i < len(common_words) - 1:
            file.write('\n')
    logging.info(f"Saving common words to {common_words_file_path}")

