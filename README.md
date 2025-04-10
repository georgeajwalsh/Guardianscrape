
# Webscrape of Guardian Data [Output 1]


1.Imports:requests allows us to make requests to websites,import pandas as pd is useful for working with tables,import random allows us to generate random items,import os is useful for interacting with computer, from datetime import datetime,timedelta gives us access for tools for working with dates and times.

`import requests
import pandas as pd
import random
import os
from datetime import datetime, timedelta`

2.Setup: API_KEY is the personal key that gives us access to the guardians content API, BASE_URL is the web adress we will be pulling data from, FILE_NAME is where we will be saving the data we collect and NUM_WEEKS specifies how many weeks of data we will look at. 

`API_KEY = "998344a2-04a1-4410-9d53-1490cfa2e9d2"
BASE_URL = "https://content.guardianapis.com/search"
FILE_NAME = "guardian_articles.csv"
NUM_WEEKS = 580`

3.Date generation: the start_date finds todays date, dates creates a list of dates one per week working backwards from todays dates fomatting as "2023-06-30".

`start_date = datetime.today()
dates = [(start_date - timedelta(weeks=i)).strftime("%Y-%m-%d") for i in range(NUM_WEEKS)]`

4.Check for previous data/create dataframe: os.path.exists check if the FILE_NAME previouslyb specified already exists on our computer if not it creates an empty dataframe with the relevant columns.

`if os.path.exists(FILE_NAME):
    df = pd.read_csv(FILE_NAME)
else:
    df = pd.DataFrame(columns=["headline", "publication_date", "url"])`

5.Begins a loop that iterates through each date in the dates list

``for week_date in dates:`

6.Creates a dictionary for the API request containing the authentication key, a from-date to to-date both set to the same value given looking at specific day, then requests headline field in response and limits to 10 articles per day

` params = {
        "api-key": API_KEY,
        "from-date": week_date,
        "to-date": week_date,
        "show-fields": "headline",
        "page-size": 10,  
    }`

7.Makes a GET request to the API using the base URL and passes the parameters dictionary storing response in response variable 

`response = requests.get(BASE_URL, params=params)`

8.Checks if the request was succesful

`if response.status_code == 200:`

9.Converts the succesful response to JSON format and stores parsed data in data variable

`data = response.json()`

10.Extracts the list of articles from the nested JSON strucutre and accesses 'response' dictionary, then accesses 'results' list within it then stores in articles variable

`articles = data["response"]["results"]`

11.Checks if any articles were found for this date

`if articles:`

12.Randomly selects one article from the available articles

`selected_article = random.choice(articles)`

13.Creates a dictionary with the headline,publication date and URL

`article_data = {
                "headline": selected_article["webTitle"],
                "publication_date": selected_article["webPublicationDate"],
                "url": selected_article["webUrl"],
            }`

14.Concatenates the new article data to existing dataframe df using ignore_index to maintain clear index

`df = pd.concat([df, pd.DataFrame([article_data])], ignore_index=True)`

15.Saves the updates dataframe to CSV file using index=False to prevent saving row numbers

`df.to_csv(FILE_NAME, index=False)`

16.This prints a success message showinf the data processed and the headline that was saved

`print(f"Saved article from {week_date}: {article_data['headline']}")`

17.Handles the case where no articles were found for the date

`else:
     print(f"No articles found for {week_date}.")`

18.Handles the case where API request failed showing the date that failed and the HTTP error code

`else:
     print(f"Failed to fetch data for {week_date}: {response.status_code}")`

19.Final message indicates all dates have been processed

`print(" Data collection complete!")`

# Import of GDP data [OUTPUT 2]


1.This calls the GDP.csv by locating it within my file directory skipping 4 rows as these are empty saving to df_gdp.

`df_gdp = pd.read_csv('/Users/georgewalsh/Desktop/API_NY/GDP.csv', skiprows=4)  # Skip the first 4 rows which contain metadata`

2.df_gdp_cleaned calls only the collumns of interest.

`df_gdp_cleaned = df_gdp[['Country Name','2014','2015','2016','2017','2018','2019', '2020', '2021', '2022','2023']]`

3.gdp_columns is all the columns containing data of interest we then convert all these values to numeric

`gdp_columns = ['2014','2015','2016','2017','2018','2019', '2020', '2021', '2022', '2023']
df_gdp_cleaned[gdp_columns] = df_gdp_cleaned[gdp_columns].apply(pd.to_numeric, errors='coerce')  # Convert values`

4.We then create a new column total GDP which summarises the data in all our columns of interest.

`df_gdp_cleaned['Total GDP'] = df_gdp_cleaned[gdp_columns].sum(axis=1)`

5.Finnally we test the success of our table manipulation by calling the first 5 lines of our dataframe.

`df_gdp_cleaned.head()`

# Visualisation country mentions against total GDP [OUTPUT 3]


1.Imports given we have already imported the majority of packages only need need import lowess this function is used for statsmodels in our case its for a smoothes line of best fit.

`from statsmodels.nonparametric.smoothers_lowess import lowess`

2.Target countries specifies the list of countries we will be performing analysis on.

`target_countries = [
    'United States', 'China', 'Japan', 'Germany', 
    'India', 'United Kingdom', 'France', 'Italy',
    'Canada', 'Brazil', 'Russia', 'South Korea'
    ]`

3.Country variants ensure we account for varations of the words in our hedline to minimise the number of articles discussuing these countries that are missed.

`country_variants = {
    'United States': ['United States', 'USA', 'US', 'America'],
    'United Kingdom': ['United Kingdom', 'UK', 'Britain'],
    'China': ['China'],
    'Japan': ['Japan'],
    'Germany': ['Germany'],
    'India': ['India'],
    'France': ['France'],
    'Italy': ['Italy'],
    'Canada': ['Canada'],
    'Brazil': ['Brazil'],
    'Russia': ['Russia'],
    'South Korea': ['South Korea']
}`

4.This creates an empty disctionary to store our mentions. For each country and it variants ot will create a regex pattern to math whole words only and counts how many headlines contain each variant and sums across all variants in the country storing the total in a dictionary.

`country_mentions = {}
for country, variants in country_variants.items():
    total = 0
    for variant in variants:
        pattern = r'\b' + re.escape(variant) + r'\b'
        count = df['headline'].str.contains(pattern, case=False, regex=True).sum()
        total += count
    country_mentions[country] = total`

5.Creates a list of mention counts in the same order as target_countries.

`mentions_counts = [country_mentions[country] for country in target_countries]`

6.This filters the GDP dataframe to only include our countries of interest and also converts country names to a catagorical variable with our specified ordr and sorts the dataframe to match our target order.

`df_filtered = df_gdp_cleaned[df_gdp_cleaned['Country Name'].isin(target_countries)]
df_filtered['Country Name'] = pd.Categorical(
    df_filtered['Country Name'], 
    categories=target_countries,
    ordered=True
    )
df_filtered = df_filtered.sort_values('Country Name')`

7.Creates a figure containing both our subplots.

`plt.figure(figsize=(14, 10))`

8.For our first plot this creates the top plot making a bar chart of mention counts using distinct colours to aid the ability to distinguish between countries in my visualisation.

`plt.subplot(2, 1, 1)
bars = plt.bar(target_countries, mentions_counts, color=plt.cm.tab20.colors[:12])`

9.This adds counnt labels to each bar by iterating through each bar in the chart retreiving the value on the y axis and then adds text to the specific coordinate such that it finds the left position of the bar then adds half the width of teh bar for central top of each bar.

`for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')`

10. This creates a smoothed red trend line through the bar heights.

`mentions_smoothed = lowess(mentions_counts, np.arange(len(target_countries)), frac=0.3)
plt.plot(target_countries, mentions_smoothed[:, 1], color='red', lw=2, label='Trend Line')`

11.This is general formatting such that it adds a title, rotates our x-ticks, adds grid lines abd a legend. 

`plt.title('Country Mentions in Headlines (Top 12 Economies)')
plt.ylabel('Number of Mentions')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.legend()`

12. This creates a second subplot in the bottom plot this creates a bar chart using the viridis colourmap.

`plt.subplot(2, 1, 2)
sns.barplot(x='Country Name', y='Total GDP', data=df_filtered, palette='viridis', order=target_countries)`

13. General formatting of our second plot

`plt.xlabel('Country')
plt.ylabel('Total GDP in Trillions(in USD)')
plt.title('Total GDP of Top 12 Economies')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)`

14.Final formatting that ensures propere spacing between subplots and displays the final visualisation

`plt.legend()
plt.tight_layout()
plt.show()`

# War datafarme import [OUTPUT 3]


1.Utalises the pandas function read the csv into the df_war dataframe skipping the first 4 rows off the file 

`df_war =pd.read_csv('/Users/georgewalsh/Documents/battle/battledata.csv',skiprows=4)`

2.This creates a new dataframe which only stores our columns of interest 

`df_war_cleaned = df_war[['Country Name','2014','2015','2016','2017','2018','2019', '2020', '2021', '2022','2023']]`

3.This creates a list of years from 2014 to 2023 storing as strings

`year_columns = [str(year) for year in range(2014, 2024)]`

4.This adds a new total column to df_war_cleaned by summing all the year columns

`df_war_cleaned['Total'] = df_war_cleaned[year_columns].sum(axis=1)`

5. THis verifies it works by calling the first 5 rows

`df_war_cleaned.head()`

# Visualisation of total deaths by war against total war mentions[OUPUT 4]


1. This first defines our countries of interest

`key_countries = ['Ukraine', 'Russia', 'United States', 'Sudan', 'United Kingdom',
                 'Afghanistan', 'Ethiopia', 'Iraq']`

2.This accounts for all variations of the word 

`country_variations = {
    'Ukraine': ['ukraine'],
    'Russia': ['russia'],
    'Iraq': ['iraq'],
    'United Kingdom': ['united kingdom', 'uk'],
    'United States': ['united states', 'us', 'usa'],
    'Ethiopia': ['ethiopia'],
    'Afghanistan': ['afghanistan'],
    'Sudan': ['sudan']
}`

3. This ensures our headlines are stored in lowercase

`df['headline_lower'] = df['headline'].str.lower()`

4. This creates an empty dictionary

`mention_counts = {}`

5(a).This loops throug each country and its variations

`for country, variations in country_variations.items():`

 (b) this creates a boolean mask for headlines containing "war" or "conflict"

 `mask = df['headline_lower'].str.contains(r'\b(war|conflict)\b', case=False)`

 (c) This combines with another mask checking for any country variation in the text

 `country_mask = mask & df['headline_lower'].apply(
        lambda text: any(variant in text for variant in variations)`

 (d) This counts the matches and stores in a dictionary

 `mention_counts[country] = country_mask.sum()`

6.This converts the dictionary to a DataFrame with columns Country abd Mentions

`mentions_df = pd.DataFrame(list(mention_counts.items()), columns=['Country', 'Mentions'])`

7.This filters the war deaths data to only include our key countries

`df_war_subset = df_war_cleaned[df_war_cleaned['Country Name'].isin(key_countries)]`

8.This will sort our dataframe by total deaths in descending order

`df_war_subset = df_war_subset.sort_values('Total', ascending=False)`

9. This creates a 2 row, 1 column figure and sets the figure size to 12x12 inches

`fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))`

10. THis plots the first total deaths plot in the colour sky blue seperating by country

`bars1 = ax1.bar(df_war_subset['Country Name'], df_war_subset['Total'], color='skyblue')`

11.This formats the figure adding a title,y axis label and faint dotted grid lines on y-axis

`ax1.set_title('Total Deaths (2014–2023)', fontsize=14)
ax1.set_ylabel('Total Deaths')
ax1.grid(axis='y', linestyle=':', alpha=0.5)`

12. This adds formatted value labels to each bar using the same method as before but adding thousand separators

`for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', 
             ha='center', va='bottom', fontsize=10)`

13.This plots our second plot mentions reordering mentions data to match deaths plot order

`mentions_df = mentions_df[mentions_df['Country'].isin(key_countries)]
mentions_df = mentions_df.set_index('Country').loc[df_war_subset['Country Name']].reset_index()`

14.Prepare's data for smoothing by extracting x and y values

`x_vals = np.arange(len(mentions_df))
y_vals = mentions_df['Mentions'].values`

15. This creates a smooth trend line using spline nterpolation only if there are enough data points plotted as a dash line

`if len(x_vals) > 2:
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    spline = make_interp_spline(x_vals, y_vals, k=2)  # you can adjust 'k' for curve smoothness
    y_smooth = spline(x_smooth)
    ax2.plot(x_smooth, y_smooth, color='darkred', linestyle='--', linewidth=2, label='Smoothed Trend')`

16.This creates  bar plot of mentions in cornflower blue 

`bars2 = ax2.bar(mentions_df['Country'], mentions_df['Mentions'], color='cornflowerblue')`

17.This formats our graph adding titles, grid lines on y axis and rotating labels for readability 

`ax2.set_title('Mentions of "War" or "Conflict" in Headlines', fontsize=14)
ax2.set_ylabel('Number of Mentions')
ax2.set_xticklabels(mentions_df['Country'], rotation=30)
ax2.grid(axis='y', linestyle=':', alpha=0.5)`

18. This adds value lavels above bars simular to first plot

`for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height}', 
             ha='center', va='bottom', fontsize=10)`

19.Final formatting

`plt.tight_layout()
plt.show()`

# Political party data import [OUTPUT 5]


1.This reads the excel file indicating the headers are in row 9

`df_pol = pd.read_excel('/Users/georgewalsh/Documents/pivottablefull.xlsx', header=8)`

2.This filters the data to only include rows where 'Data' column equals 'Sum of Vote' and then only selecting the 'Party' column and and the years '2015','2017' and '2019' then creating a new dataframe with this subset of data 

`df_vote = df_pol[df_pol['Data'] == 'Sum of Vote'][['Party', 2015, 2017, 2019]]`

3.This creates a new column 'Total Votes' which calculates the sum of votes across the three election years in a row-wise summation

`df_vote['Total Votes'] = df_vote[[2015, 2017, 2019]].sum(axis=1)`

4.This calculates the grand total of all votes across parties

`total_all_parties = df_vote['Total Votes'].sum()`

5.Creates a new column 'Percentage' which calculates each parties percentage by total vote 

`df_vote['Percentage'] = (df_vote['Total Votes'] / total_all_parties) * 100`

6.Replaces abbreviated party names with their full names using a dictionary mapping

`df_vote['Party'] = df_vote['Party'].replace({
    'CON': 'Conservative',
    'LAB': 'Labour',
    'LIB': 'Lib Dem',
    'NAT': 'Scotish National Party'
})`

7.Filters out usinf '~' logic (not) and isin to remove 'MIN' and 'OTH' parties

`df_vote = df_vote[~df_vote['Party'].isin(['MIN', 'OTH'])]`

8.This orders the dataframe by percentage column in order largest to smallest percentage

`df_vote = df_vote.sort_values('Percentage', ascending=False)`


# Visualisation of party votes against party mentions [OUTPUT 6]


1.Creates a new figure with dimensions 14x7 inches

`plt.figure(figsize=(14, 7))`

2.This creates a bar chart where the x-axis is df_vote['Party'] and the y-axis is df_vote['Percentage'] and then setting specific hex codes for the each party

`bars_vote = plt.bar(df_vote['Party'], df_vote['Percentage'], color=[
    '#0087DC', '#E4003B', '#FAA61A', '#3F8428','#6D3177','#999999'
])`

3.Loops through each bar in the chart and gets the height of each bar 

`for bar in bars_vote:
    height = bar.get_height()`

4.This adds text labels above each bar simular to done previously

` plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom')`

5.This sets the format of the graph with a title with 20 pixel padding below it, a y-label,sets y-axis limits from 0 to 5%  above the max percentage value ,adds faint horizontal grid lines and adjusts subplot params to prevent overlapping then displays the figure.

`plt.title('UK General Elections Total Vote Share by Party', pad=20)
plt.ylabel('Percentage of Votes (%)')
plt.ylim(0, df_vote['Percentage'].max() + 5)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()`

6.Creates a dictionary to count mentions which defaults to 0 for new keys

`mention_counts = defaultdict(int)`

7.This loops through every headline in the dataframe and creates a set to track party mentions in current headline 

`for headline in df['headline']:
    found_parties = set()`

8.Checks if any party's regex pattern matches the headline and adds matching parties to the set

` for party, pattern in patterns.items():
        if pattern.search(headline):
            found_parties.add(party)`

9.Increments mention count for each mentioned party

`for party in found_parties:
        mention_counts[party] += 1`

10.This converts the dictionary to a dataframe with party names as index 

`count_df = pd.DataFrame.from_dict(mention_counts, orient='index', columns=['Count'])`

11.This calculates the total mentions across all parties and then adds a percentage column as a proportion of total mentions 

`total_mentions = count_df['Count'].sum()
count_df['Percentage'] = (count_df['Count'] / total_mentions) * 100`

12.Sorts by percentage in descending order

`count_df = count_df.sort_values('Percentage', ascending=False)`

13.Creates new figure the same size as previous chart 

`plt.figure(figsize=(14, 7))`

14. Defines colour mapping to the same as previous chart

`colors = {
    'Conservative': '#0087DC',
    'Labour': '#E4003B',
    'Liberal Democrat': '#FAA61A',
    'Scotish National Party': '#3F8428'  
}`

15.This creates a bar  chart using party names as x-values and percentage values as heights using the colour mapping from the dictionary

`bars_mentions = plt.bar(count_df.index, count_df['Percentage'], 
                       color=[colors[p] for p in count_df.index])`

16.Adds percentage labels above bars (same format as previous)

`for bar in bars_mentions:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%',
             ha='center', va='bottom', fontsize=10)`

17.Formats the table setting the title,adding labels, grid,layout then displays the figure

`plt.title('Political Party Mentions in Headlines (% of Total Mentions)', pad=20)
plt.ylabel('Percentage of Mentions (%)')
plt.ylim(0, count_df['Percentage'].max() + 5)
plt.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.show()`

