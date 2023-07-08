import urllib.request
import os
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import math
from nltk.stem import WordNetLemmatizer
from collections import defaultdict


df = pd.read_csv('appsreviewDataset.csv', encoding='utf-8')
df.head(3)

# init Objects
tokenizer=RegexpTokenizer(r'\w+')
en_stopwords=set(stopwords.words('english'))
ps=PorterStemmer()
def getStemmedReview(review):
    review=review.lower()
    review=review.replace("<br /><br />"," ")
    #Tokenize
    tokens=tokenizer.tokenize(review)
    new_tokens=[token for token in tokens if token not in  en_stopwords]
    stemmed_tokens=[ps.stem(token) for token in new_tokens]
    clean_review=' '.join(stemmed_tokens)
    return clean_review

#Split train and test 70:30
df['review'].apply(getStemmedReview)
X_train = df.loc[:21000, 'review'].values
y_train = df.loc[:21000, 'sentiment'].values
X_test = df.loc[9000:, 'review'].values
y_test = df.loc[9000:, 'sentiment'].values

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8',
 decode_error='ignore')
vectorizer.fit(X_train)
X_train=vectorizer.transform(X_train)
X_test=vectorizer.transform(X_test)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(X_train,y_train)
print("Score on training data is: "+str(model.score(X_train,y_train)))
print("Score on testing data is: "+str(model.score(X_test,y_test)))

lemmatizer = WordNetLemmatizer()

# word_counts[word][0] = occurrences of word in negative reviews
# word_counts[word][1] = occurrences of word in positive reviews
word_counts = defaultdict(lambda: [0, 0]) # returns [0, 0] by default if the key does not exist

STOP_WORDS = stopwords.words('english')

tokenizer = RegexpTokenizer(r'\w+')

sentiment = list(df['sentiment'])

done =  0

total_positive_words = 0
total_negative_words = 0

# keep track of the number of positive and negative reviews (prior probabilities)
total_positive_reviews = 0
total_negative_reviews = 0


for i, review in enumerate(list(df['review'])):
    if sentiment[i] == 'positive':
        total_positive_reviews += 1
    else:
        total_negative_reviews += 1
    
    for token in tokenizer.tokenize(review):
        token = token.lower()
        token = lemmatizer.lemmatize(token)
        if token not in STOP_WORDS:
            if sentiment[i] == 'positive':
                word_counts[token][1] += 1
                total_positive_words += 1
            else:
                word_counts[token][0] += 1
                total_negative_words += 1


word_counts = sorted(word_counts.items(),  key=lambda x : x[1][0] + x[1][1], reverse=True)[:5000]

word_counts = defaultdict(lambda: [0, 0], word_counts)

def calculate_word_probability(word, sentiment):
    if sentiment == 'positive':
        return math.log((word_counts[word][1] + 1) / (total_positive_words + 5000))
    else:
        return math.log((word_counts[word][0] + 1) / (total_negative_words + 5000))


def calculate_review_probability(review, sentiment):
    if sentiment == 'positive':
        probability = math.log(total_positive_reviews / len(df))
    else:
        probability = math.log(total_negative_reviews / len(df))
    
    for token in tokenizer.tokenize(review):
        token = token.lower()
        token = lemmatizer.lemmatize(token)
        if token not in STOP_WORDS:
            probability += calculate_word_probability(token, sentiment)
    return probability


def predict(review):
    if calculate_review_probability(review, 'positive') > calculate_review_probability(review, 'negative'):
        return 'positive'
    else:
        return 'negative'
