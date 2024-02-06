# Final project epam ds course

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

You can find brief comparation of this methods in 

# MLE part

## Initial preparations

Initially you need to run 

`python3 ./src/data_loader.py`

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
docker cp <container_id>:/app/outputs/models ./outputs/

docker cp <container_id>:/app/outputs/models ./outputs/
```


Replace `<container_id>` with your running Docker container ID.