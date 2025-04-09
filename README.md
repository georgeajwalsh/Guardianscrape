
WEB SCRAPE-OUTPUT 1

1.imports: 
requests allows us to make requests to websites,import pandas as pd is useful for working with tables,import random allows us to generate random items,import os is useful for interacting with computer, from datetime import datetime,timedelta gives us access for tools for working with dates and times

`import requests
import pandas as pd
import random
import os
from datetime import datetime, timedelta`

2.setup: 
API_KEY is the personal key that gives us access to the guardians content API, BASE_URL is the web adress we will be pulling data from, FILE_NAME is where we will be saving the data we collect and NUM_WEEKS specifies how many weeks of data we will look at 

`API_KEY = "998344a2-04a1-4410-9d53-1490cfa2e9d2"
BASE_URL = "https://content.guardianapis.com/search"
FILE_NAME = "guardian_articles.csv"
NUM_WEEKS = 580`

3.date generation: 
the start_date finds todays date, dates creates a list of dates one per week working backwards from todays dates fomatting as "2023-06-30"

`start_date = datetime.today()
dates = [(start_date - timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(NUM_WEEKS)]`

4.check for previous data/create dataframe 
os.path.exists check if the FILE_NAME previouslyb specified already exists on our computer if not it creates an empty dataframe with the relevant columns

`if os.path.exists(FILE_NAME):
    df = pd.read_csv(FILE_NAME)
else:
    df = pd.DataFrame(columns=["headline", "publication_date", "url"])`

5.fetch articles for each week
first for week_date in dates is a forloop iterating through each date, params sets up the API paramaters for the given date such that it i for the given date showing headlines and picks up to 10 articles per day. Response then requests the guardian API for our given parameters if response.status_code then checks if the request went through with 200 meaning 'OK' if this condition is met this will convert the respone into readable data grabbing the relevant data. the following if function if articles checks if we got any articles for that given date if so it keeps going such that selected_article will pick a random article. aricle_data creates a small dictionary with the atricle title, publication date and URL where it can be read. df= saves our new article to the existing table df and then df.to_csv saves the whole updated file back to CSV file.. The rest of the code is used to adress issues with colection such that the first print gives us a little sucessa message to see that something was saved for that date with the follwoing print in the else command telling us if nothing came up for given day and the following else will tell us it failed to fetch with the status code so I can see what went wrong. FIaly wants the loop is complete I print a data collection complete message

`for week_date in dates:
    params = {
        "api-key": API_KEY,
        "from-date": week_date,
        "to-date": week_date,
        "show-fields": "headline",
        "page-size": 10,  # Get up to 10 articles from the date
    }

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        articles = data["response"]["results"]
        
        if articles:
            selected_article = random.choice(articles)  # Pick one randomly
            article_data = {
                "headline": selected_article["webTitle"],
                "publication_date": selected_article["webPublicationDate"],
                "url": selected_article["webUrl"],
            }

            # Append new data and save
            df = pd.concat([df, pd.DataFrame([article_data])], ignore_index=True)
            df.to_csv(FILE_NAME, index=False)

            print(f"Saved article from {week_date}: {article_data['headline']}")
        else:
            print(f"No articles found for {week_date}.")
    else:
        print(f"Failed to fetch data for {week_date}: {response.status_code}")

print(" Data collection complete!")`
