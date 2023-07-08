#Running
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

from collections import defaultdict
import math

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd

print('Load and train dataset...')
df = pd.read_csv('preprocessDataset2.csv')

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

contraction_map = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}

import re
def preprocess(value):
    #convert the review to lower case
    value=value.lower()
    #contraction map
    value = ' '.join([contraction_map[t] if t in contraction_map else t for t in value.split(" ")])
    #remove all urls
    value=re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', value)
    #remove all @username
    value=re.sub('@[^\s]+', '', value)
    #convert #topic to "topic"
    value=re.sub(r'#([^A-Za-z ]+)', r'\1', value)
    #remove special characters and numbers
    value=re.sub('[^A-Za-z ]+', '', value)
    return value

#Load Input and Predict
print('Classifying all reviews...')

# load excel with its path
df5 = pd.read_csv('playstorescrapping.csv')

countpositive=0
countnegative=0

review = []
sentiment = []

getcontent=df5['content']
for i, content in enumerate(list(df5['content'])):
    preprocesscontent=preprocess(getcontent[i])
    result=predict(preprocesscontent)

    review.append(preprocesscontent)
    sentiment.append(result)

    if result == "positive":
        countpositive=countpositive+1
    else:
        countnegative=countnegative+1

data = {'review': review, 'sentiment': sentiment}

print('Exporting classify result to excel...\n')
df3 = pd.DataFrame(data)
df3.to_csv("processresult.csv", index=False)


#df6 = pd.read_csv('processresult.csv')
correct  = 0
incorrect = 0
sentiments = list(df['sentiment'])
for i, text in enumerate(list(df['review'])):
    #print(text)
    if predict(text) == sentiments[i]:
        correct += 1
    else:
        incorrect += 1

print('Result:\n')
print('Correct: ',correct)
print('Incorrect: ',incorrect)
accuracy=(correct / (correct + incorrect))*100
print('Model accuracy: ','%.2f' % accuracy,'%')

print('Analysis result: ')
print('Positive sentiment : ',countpositive)
print('Negative sentiment : ',countnegative)

if countpositive>countnegative:
    print('Application review: Good')
else:
    print('Application review: Bad')
        



