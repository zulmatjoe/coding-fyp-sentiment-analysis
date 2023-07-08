import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

df = pd.read_csv('all_tickets.csv')

df.sample(3).T

#df.ticket_type.value_counts()
#df.category.value_counts()
#df.business_service.value_counts()
#df.urgency.value_counts()
#df.impact.value_counts()

#Preprocess
count_vec = CountVectorizer()
bow = count_vec.fit_transform(df['text'])
bow = np.array(bow.todense())

#Train and test 
X = bow
y = df['ticket_type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model = MultinomialNB().fit(X_train, y_train)

#Model performance

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred, average="macro"))

print(classification_report(y_test, y_pred))

