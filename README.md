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
- removing stopwords (most common words of english language, such as 'the', 'of' etc) 

Then I remove words, that contains 3 or less characters.

In common_words notebook you can see initialization of common_words, by which I mean most used words, common for both 'positive' and 'negative' reviews. I calculate most used words independently and then find intersection. As you can see, both datasets contains a lot of words 'film', 'cinema', which is quite obvious. But, on another hand, they contains a lot of words 'good'. It's contrintuitive from the first sight, but we understand, that, for example, 'negative' reviews may contain phrases like 'I thought it's a good movie'/'This is not a good movie' etc.

Firstly, you need to find common words of preprocessed data, and then to make preprocess itself. The difference is as follows: when I find common words, I work with text only. In preprocessing part I make tokenization after converting to lowercases. I think it's a good approach, as we delete stopwords and commonwords as tokens, and it means that more words will be deleted.

This notebooks code is in save_common_words.py and preprocess.py.

## Lemmatization vs Stemming

You can find brief comparation of this methods in lemmatization_vs_stemming notebook.

Quite commonly, I chose lemmatization over stemming.
There are several reasons for that:

-
-
-


## Bag of words vs TF-IDF

In bow_vs_tfidf notebook you can find comparation of bow and tf-idf vectorizations. It's notable, that bag of words creates vectors with around 100 000 dimensions and this vectors calculates for large amounts of time obviously. 

I compared tf-idf with different max_features parameters and bag of words. As you can see, max_features = 2000 allows us to predict outcome with almost same accuracy, recall and f1 score (with decrease of 1% in the worst cases). For decision tree tf-idf got even better results. 

For my calculations in train.py I use tf-idf with max_features = 2000. You can change this parameter (and min_df and max_df also) in settings.json.

In common words notebook I found, that there are a lot of words, that are presented in only one copy (unique words) in reviews. We can't really predict impact of this words on outcome. min_df = 7 removes this rare words. max_df = 1.0 is a default value.

## Finding best model

You can find comparation of the models in find_best_model notebok.

# MLE part

## Initial preparations

Initially you need to run 

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