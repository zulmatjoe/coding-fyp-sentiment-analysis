#Scraping Reviews...

#IMPORT LIBRARIES
from google_play_scraper import Sort, reviews_all, reviews, app
import pandas as pd
import numpy as np

print('Enter apps address: ')
appsName=input()
print(appsName)

#SCRAPE COUNT REVIEWS
stringUrl=appsName
result, continuation_token = reviews(
    stringUrl,
    lang='en',                # defaults to 'en'
    country='us',             # defaults to 'us'
    sort=Sort.MOST_RELEVANT,  # defaults to Sort.MOST_RELEVANT you can use Sort.NEWEST to get newst reviews
    count=3000,              # defaults to 100
    filter_score_with=None    # defaults to None(means all score) Use 1 or 2 or 3 or 4 or 5 to select certain score
)

#SCRAPING RESULT
scrapeddata = pd.DataFrame(np.array(result),columns=['review'])

scrapeddata = scrapeddata.join(pd.DataFrame(scrapeddata.pop('review').tolist()))

scrapeddata.head()

#FILTERING SCRAPPED DATA
scrapeddata[['content']].head() 

#SORT TO DATE
scrappeddata1 = scrapeddata[['content']]
#sorteddata = scrappeddata1.sort_values(by='content', ascending=True) #Sort by Newest, change to True if you want to sort by Oldest.
scrappeddata1.head()
 
 #IMPORT TO EXCEL
scrappeddata1.to_csv("playstorescrapping.csv", index = False)
#scrappeddata1.to_excel("playstorescrapping.xlsx", index = False)  #Save the file as CSV , to download: click the folder icon on the left. the csv file should be there.

print('Successfully scraped the reviews')