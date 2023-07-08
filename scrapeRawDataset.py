#Scraping Reviews...

#IMPORT LIBRARIES
from google_play_scraper import Sort, reviews_all, reviews, app
import pandas as pd
import numpy as np

result, continuation_token = reviews(
    'com.instagram.android',
    lang='en',                # defaults to 'en'
    country='us',             # defaults to 'us'
    sort=Sort.MOST_RELEVANT,  # defaults to Sort.MOST_RELEVANT you can use Sort.NEWEST to get newst reviews
    count=15000,               # defaults to 100
    filter_score_with=1    # defaults to None(means all score) Use 1 or 2 or 3 or 4 or 5 to select certain score
)

scrapeddata = pd.DataFrame(np.array(result),columns=['review'])

scrapeddata = scrapeddata.join(pd.DataFrame(scrapeddata.pop('review').tolist()))

scrapeddata.head()

scrappeddata1 = scrapeddata[['content','score']]
sorteddata = scrappeddata1.sort_values(by='score', ascending=True) #Sort by Newest, change to True if you want to sort by Oldest.
sorteddata.head()

sorteddata.to_csv("rawDatasetNegative.csv", index = False)

print("Successfully scraped the raw Dataset")