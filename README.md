# Final project epam ds course

All of the following were tested via macos environment

# DS part

## EDA
EDA is presented in EDA and common_words notebooks

From EDA notebook we can see that our train data has no missing values and it is a balanced dataset (equal number of positive and negative reviews). In notebooks you can see, that data preprocessing mainly consists of data cleaning.

As parts of data cleaning I use

- removing url's
- removing html tags
- removing non alphabetic symbols
- converting all characters to lowercases
- removing stopwords 

Let me describe removing stopwords in details. Stopwords are common words include articles, conjuntions, prepositions and pronouns (such as 'the', 'of', 'is', 'at', 'which' etc). Removing stopwords helps us to

- Remove some words with no meaning
- Reduce dataset size
- Improve model performance
- Remove noise from data

I used nlkt.stopwords to identify stopwords.

Then I remove words, that contains 3 or less characters.

In common_words notebook you can see initialization of common_words, by which I mean most used words, common for both 'positive' and 'negative' reviews. I calculate most used words independently and then find intersection. As you can see, both datasets contains a lot of words 'film', 'cinema', which is quite obvious. But, on the other hand, they contains a lot of words 'good'. It's contrintuitive from the first sight, but we understand, that, for example, 'negative' reviews may contain phrases like 'I thought it's a good movie'/'This is not a good movie' etc.

Firstly, you need to find common words of preprocessed data, and then to make preprocess itself. The difference is as follows: when I find common words, I work with text only. In preprocessing part I make tokenization after converting to lowercases. I think it's a good approach, as we delete stopwords and commonwords as tokens, and it means that more words will be deleted.

This notebooks code is in save_common_words.py and preprocess.py.

## Lemmatization vs Stemming

You can find brief comparison of this methods in lemmatization_vs_stemming notebook.

Quite commonly, I chose lemmatization over stemming.
There are several reasons for that:

- Lemmatization considers the context and uses morphological analysis of words. It removes inflectional endings only. It leads to better semantic analysis. Stemming may lose actual meaning of words (see short examples in notebook)
- Overall, lemmatization is more accurate than stemming, because stemming simply chop off the ends of words. It can lead to incorrect generalizations and provide us with more false positives or negatives. On the other hand, lemmatization uses a vocabulary and morphological analysis, reducing the chances of errors
- Lemmatization is better at dealing with irregular words. For example, with lemmatization 'went' transforms to 'go'
- Lemmatization provides us with better readability (see notebook examples)


## Bag of words vs TF-IDF

In bow_vs_tfidf notebook you can find comparation of bow and tf-idf vectorizations. It's notable, that bag of words creates vectors with around 100 000 dimensions and this vectors calculates for large amounts of time obviously. 

I compared tf-idf with different max_features parameters and bag of words. As you can see, max_features = 2000 allows us to predict outcome with almost same accuracy, recall and f1 score (with decrease of 1% in the worst cases). For decision tree tf-idf got even better results. 

For my calculations in train.py I use tf-idf with max_features = 2000. You can change this parameter (and min_df and max_df also) in settings.json.

In common words notebook I found, that there are a lot of words, that are presented in only one copy (unique words) in reviews. We can't really predict impact of this words on outcome. min_df = 7 removes this rare words. max_df = 0.8 (default value 1.0). It cut's off words, that appears too often. I already deleted most frequently common words. With help of this parameter, it cuts off even more frequently used words. From my experience values between 0.8 and 1.0 have no impact on final metrics. I chose 0.8 value, because for me it looks reasonable to exclude most frequently used words. We may not use common_words function and just use max_df, but my common_words are part of EDA also. common words notebook shows this words in train data. However, there will be no rare words in top 2000 features anyway.

Summing up, tf-idf with max_features = 2000, min_df = 7, max_df = 0.8 have almost same metrics, but a lot higher computational performance.

## Preprocessing pipeline

As was written above, in preprocessing pipeline I make tokenization after converting to lowercases and then I remove stopwords, small words. For tokenization I use nltk.tokenize.word_tokenize. It is a common package for tokenization. 

## Finding best model

In my research I got the following results for metrics on inference data:

- Naive Bayes.
Accuracy: 82.5%.
Recall: 83.6%.
F1-score: 82.7%.

- Decision tree.
Accuracy: 75.5%.
Recall: 70.8%.
F1-score: 74.3%.

- Logistic regression.
Accuracy: 88.0%.
Recall: 87.6%.
F1-score: 88.0%.

You can find comparation of the models in find_best_model notebok. I found, that in my case the best model is Logistic Regression. It shows the best metrics and it evaluates faster than others. Decision tree and Naive bayes got less than 85% in my case. Maybe this can be reduced with another vectorization parameters, but it's not obvious that this models may show metrics much higher than logistic regression. As was written previously, I think, that if we want all metrics 95+ %, we need to make neuron network with huge amount of neurons, but even evaluation training of such network could take days.


## Python scripts

In my python scripts I got 87.2% mean cross validation accuracy on train dataset. For inference dataset I got:

- accuracy = 87.5%
- recall = 87.1%
- f1 score = 87.4%

This results are a little bit worse, than in find_best_model notebook. The cause of this difference is not clear for me.

I think, my model gives quite a good performance. It may be improved with help of using neuron networks or maybe by better feature engineering, i.e. finding more common_words, using different min_df and max_df for tf-idf etc. However, this approaches requires a lot of more research. As the first step, I suppose my model is good.

## Potential business applications and value for business


- Film recommendation system. My sentiment classification model may help to increase accuracy of recommendations on some film aggregators. With understanding the sentiment of reviews, users can be better alligned with their preferences (i.e. if user has positive review to a film, he will see movies of same director/main actor etc in his recommendations).

- This model can help film aggregators to choose popular films and to create some pages with selectioned movies, like 'Best movies of 2023', 'Best horrors' etc. This can increase users time-spending on the platform.

- Marketing. Film makers, studios, distributors may want to understand, what are current trends of mass-market movies, i.e. they may want to understand, what genres are in trends. Where to invest the money?

- Content moderation. Similar to previous, but here film aggregators may want to add new films to their library, or to delete unpopular ones

- Advertisement. Companies may see the difference in people sentiments before and after promotional campaigns/ before and after merchandise etc. Also companies may realize, that their film is worth watching or not for most of the people, based on early reviews. This can help to decide, if they need to make more promotion in order to attract people or not to waste more money.



# MLE part

## Project structure

Project organizes as follows.
Data folder contains raw folder (initial data) and processed folder (preprocessed data). Models and vectorizers are saved in outputs folder (/outputs/models and /outputs/vectorizer). Final results on inference are saved in /outputs/predictions.

## Initial preparations

You may install all necessary environment via 

```python
pip3 install --no-cache-dir -r requirements.txt
```

Initially for loading data you need to run 

`python3 ./src/data_loader.py`


## Training part

Then you can create training image. You need to run Docker daemon and then 

```python
docker build -f ./src/train/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
You can run this container with 

```python
docker run -it training_image /bin/bash
```

Then you need to copy data from this container with help of 


```python
docker cp <container_id>:/app/outputs/ ./

docker cp <container_id>:/app/data/processed/ ./data/processed
```


Replace `<container_id>` with your running Docker container ID.

As alternative, you may simply run locally

```python
python3 ./src/train/save_common_words.py

python3 ./src/train/preprocess.py

python3 ./src/train/train.py
```

## Inference part

For inference docker image creation you can use 

```python
docker build -f ./src/inference/Dockerfile --build-arg settings_name=settings.json -t inference_image .
```

You can run container with 

```python
docker run -it inference_image /bin/bash 
```

Then you need to copy results to local machine with 

```python
docker cp <container_id>:/app/outputs/predictions/ ./outputs
```

Alternatively, you can use locally

```python
python3 ./src/inference/run_inference.py
```
