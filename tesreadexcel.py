import pandas as pd

# import a list of films
df = pd.read_csv('testexcel.csv')

getreview=df['review']
getsentiment=df['sentiment']
for i, review in enumerate(list(df['review'])):
    print(getreview[i])
    print(getsentiment[i])
    
