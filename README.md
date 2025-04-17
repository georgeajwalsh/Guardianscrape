# Web scrape of Guardian Data [Output 1]

1. Requests allows us to make requests to websites, import pandas as pd is useful for working with tables, import random allows us to generate random items,  import so is useful for interacting with computer, from datetime import datetime,timedelta gives us access for tools for working with dates and times.

`import requests
import pandas as pd
import random
import os
from datetime import datetime, timedelta`

2. API_KEY is the personal key that gives us access to the Guardian's content API, BASE_URL is the web address we will be pulling data from, and FILE_NAME is where we will be saving the data we collect

`API_KEY = "998344a2-04a1-4410-9d53-1490cfa2e9d2"
BASE_URL = "https://content.guardianapis.com/search"
FILE_NAME = "guardian_articles.csv"`

3. Setup: Defining the date range and number of articles to fetch

`start_year = 2014
end_year = datetime.today().year
total_years = end_year - start_year + 1
total_weeks_to_scrape = 554  # total number of weeks/articles desired
weeks_per_year = total_weeks_to_scrape // total_years`

4. First, define the function to generate a list of evenly spaced dates, set the start and end of the year, calculate the number of days in that year, determine the spacing between each article date, and then generate a list of dates spaced by step days, then format such that (YYYY-MM-DD)

`def get_evenly_spaced_dates_for_year(year, weeks):
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    days_between = (end - start).days
    step = days_between // weeks
    return [(start + timedelta(days=i * step)).strftime("%Y-%m-%d") for i in range(weeks)]`

5. Generate an empty list to store all generated dates and then loop through each year, generating the weekly dates and adding to all_dates

`all_dates = []
for year in range(start_year, end_year + 1):
    all_dates.extend(get_evenly_spaced_dates_for_year(year, weeks_per_year))`

6. Shuffle the dates so the API requests are not in chronological order to help address rate limits and bias

`random.shuffle(all_dates)`

6. This will load the existing file if it exists, otherwise it will start with a new data frame with the defined columns

`if os.path.exists(FILE_NAME):
    df = pd.read_csv(FILE_NAME)
else:
    df = pd.DataFrame(columns=["headline", "publication_date", "url"])`

7. Loops through each date

`for week_date in all_dates:`

8. Set up the parameters for the API request

`    params = {
        "api-key": API_KEY,
        "from-date": week_date,
        "to-date": week_date,
        "show-fields": "headline",
        "page-size": 10,
    }`

9. Send a GET request to the API using the above parameters

` response = requests.get(BASE_URL, params=params)`

10. Check if request was successful

`  if response.status_code == 200:`

11. Parse the JSON response and extract the list of articles

`data = response.json()
        articles = data["response"]["results"]`
12. If artcles are returned, pick one at random

`if articles:
            selected_article = random.choice(articles)`


13. Extracts the desired fields and stores them in a dictionary

`        article_data = {
                "headline": selected_article["webTitle"],
                "publication_date": selected_article["webPublicationDate"],
                "url": selected_article["webUrl"],
            }`

14. Append the new article as a row in the Data Frame

`df = pd.concat([df, pd.DataFrame([article_data])], ignore_index=True)`

15. Save the updated Data Frame to the CSV file

`df.to_csv(FILE_NAME, index=False)`

16. Prints a confirmation that the article was saved

`print(f"Saved article from {week_date}: {article_data['headline']}")`

17. If no article was found for that date, print a message

`else:
            print(f"No articles found for {week_date}.")`

18. If the request failed (non-200 status), print an error message

` else:
        print(f"Failed to fetch data for {week_date}: {response.status_code}")`

19. Once all dates are processed, it prints a completion message

`print("Data collection complete!")`

# Data Distribution [Output 2]

1.Import pandas for data manipulation, matplotlib.pyploy for plotting, seaborn for statistical plots and NumPy for numerical operations

`import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np`

2. Converts the 'publication_date' column to the correct date time format

`df['publication_date'] = pd.to_datetime(df['publication_date'])`

3. Extracts the year and month from the datetime

`df['year'] = df['publication_date'].dt.year
df['month'] = df['publication_date'].dt.month_name()`

4. Counts how many entries there are per year, then sorts years in order and then prepares labels for plotting

`year_counts = df['year'].value_counts().sort_index()
year_index = year_counts.index.astype(str)
year_values = year_counts.values
`

5. Groups data by year and month, counting entries per group to the then convert to a Data Frame with a count column

`monthly_distribution = df.groupby(['year', 'month']).size().reset_index(name='count')`

6. Ensures months appear in calendar order

monthly_distribution = df.groupby(['year', 'month']).size().reset_index(name='count')



7. Ensure consistent month order

`month_order = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]
monthly_distribution['month'] = pd.Categorical(monthly_distribution['month'], categories=month_order, ordered=True)`

8. Creates a figure with two side-by-side plots

`fig, axes = plt.subplots(1, 2, figsize=(18, 7))`

9. Chooses a colour palette with enough colours for each year

`colors = sns.color_palette("tab10", len(year_counts))`

10. Draws a bar chart on the left subplot fir each year 

`bars = axes[0].bar(
    year_index,
    year_values,
    color=colors,
    edgecolor='white',
    width=0.6,
    label='Entries'
)`

11. Adds a red smoothed trend line

`sns.lineplot(
    x=year_index,
    y=year_values,
    color='red',
    linewidth=1,
    label='Trend',
    ax=axes[0]
)`

12. Annotates the bar chart with value labels above each bar

`for bar in bars:
    height = bar.get_height()
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f'{height}',
        ha='center',
        va='bottom',
        fontsize=10
    )`

13. Adds labels, title, gridlines and removes top spines for a cleaner look

`axes[0].set_title("Entries Per Year with Trend Line", fontsize=16, weight='bold', pad=15)
axes[0].set_xlabel("Years", fontsize=12)
axes[0].set_ylabel("Number of Entries", fontsize=12)
axes[0].tick_params(axis='x', rotation=0, labelsize=10)
axes[0].tick_params(axis='y', labelsize=10)
axes[0].grid(axis='y', linestyle='--', alpha=0.4)
sns.despine(top=True, right=True, ax=axes[0])`


14. Creates a boxplot on the right subplot showing how entry counts vary by month

`sns.boxplot(
    data=monthly_distribution,
    x='month',
    y='count',
    palette='pastel',
    linewidth=1.2,
    fliersize=3,
    ax=axes[1]
)`

15. For each month, gets entry counts, computes quartiles, median and whisker and then annotates the plot with these statistics

`for i, month in enumerate(month_order):
    month_data = monthly_distribution[monthly_distribution['month'] == month]['count'].dropna()
    if len(month_data) == 0:
        continue
    q1 = np.percentile(month_data, 25)
    q3 = np.percentile(month_data, 75)
    median = np.median(month_data)
    whisker_low = month_data[month_data >= q1 - 1.5 * (q3 - q1)].min()
    whisker_high = month_data[month_data <= q3 + 1.5 * (q3 - q1)].max()
       axes[1].text(i, median + 2, f'Median: {int(median)}', ha='center', va='center', fontsize=7, color='black', weight='bold')
    axes[1].text(i, q1, f'Q1: {int(q1)}', ha='center', va='top', fontsize=8, color='darkblue')
    axes[1].text(i, q3, f'Q3: {int(q3)}', ha='center', va='bottom', fontsize=8, color='darkgreen')
    axes[1].text(i, whisker_low, f'Min: {int(whisker_low)}', ha='center', va='top', fontsize=8, color='gray')
    axes[1].text(i, whisker_high, f'Max: {int(whisker_high)}', ha='center', va='bottom', fontsize=8, color='gray')`

16. Styles the box plot similarly adding title, axis labels and grid

`axes[1].set_title("Box Plot of Monthly Entry Counts Across Years", fontsize=16, weight='bold', pad=20)
axes[1].set_xlabel("Month", fontsize=12)
axes[1].set_ylabel("Number of Entries per Year", fontsize=12)
axes[1].tick_params(axis='x', rotation=45, labelsize=10)
axes[1].tick_params(axis='y', labelsize=10)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
sns.despine(top=True, right=True, ax=axes[1])`

17. Finally, adjust layout to prevent overlapping and display the complete figure

`plt.tight_layout()
plt.show()`

# Word Cloud [Output 3]

1. Imports the Word Class and the built-in list of STOPWORDS from the word cloud library

`from wordcloud import WordCloud, STOPWORDS`

2. Merges all non-empty headlines into a single string with each headline being separated by a space

`text = " ".join(headline for headline in df['headline'].dropna())`

3. Converts the default set of stop words into a Python set and then adds additional specific common words that are not useful to our analysis

`stopwords = set(STOPWORDS)
stopwords.update(["s", "said", "mr", "mrs",
                  "says","will","happened","review",
                  "quick","U","new","crossword", "Cryptic",""
                  "day","call","year"])`

4. Creates a Word Cloud object with specified width, height, white background, the cleaned list of stop words, and a colour map (viridis)

`wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      stopwords=stopwords,
                      colormap='viridis').generate(text)`

5. Sets up a plot with size 10x5, displays the word cloud using bilinear interpolation, removed axis for cleaner look, adds a title and then renders the final word cloud using plt.show()

`plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Headlines (2013-2023)')
plt.show()`

# Premier League data import [Output 4]

1. Loads the Premier League match data from csv into our data frame

`df_footy=pd.read_csv("/Users/georgewalsh/Documents/premier-league-matches.csv")`

2. Filter our dataset to only include matches from 2015-2016 season onwards

`df_footy_filtered = df_footy[df_footy['Season_End_Year'] >= 2016]`

3. Identifies matches where the home team scored more than away teams and counts how many time as each home team won at home

`home_wins = df_footy_filtered[df_footy_filtered['HomeGoals'] > df_footy_filtered['AwayGoals']]
home_win_counts = home_wins['Home'].value_counts()`

4. Identifies matches where away team scored more goals than home and then counts how many times each away team won away from home

`away_wins = df_footy_filtered[df_footy_filtered['AwayGoals'] > df_footy_filtered['HomeGoals']]
away_win_counts = away_wins['Away'].value_counts()`

5. Adds the home and away win for each team, uses fill value to ensure any teams that only appear in one are still counted and then converts the combined totals into integers

`total_wins = home_win_counts.add(away_win_counts, fill_value=0).astype(int)`

6. Converts the total wins into a new data  frame renaming the columns to 'team' and 'wins'

`df_team_wins = total_wins.reset_index()
df_team_wins.columns = ['team', 'wins']`

7. Sorts the teams in descending order based on number of wins and then resets the index to give clean, consecutive numbering

`df_team_wins = df_team_wins.sort_values(by='wins', ascending=False).reset_index(drop=True)`

8. Filters the Guardian data frame to ensure consistency of timelines

`df_2023 = df[(df['year'] >= 2015) & (df['year'] <= 2023)]`

9. Displays the top 5 teams with the most wins from the filtered football dataset

`df_team_wins.head()`


# Premier League visualisation [Output 5]

1. Creates a dictionary called patters where each key is a football team and the value is a regex pattern that matches common variations and not case sensitive as a result of re.IGNORECASE

`patterns = {
    'Arsenal': re.compile(r'\bArsenal\b|\bGunners\b', re.IGNORECASE),
    'Aston Villa': re.compile(r'\bAston Villa\b|\bVilla\b', re.IGNORECASE),
    'Bournemouth': re.compile(r'\bBournemouth\b|\bCherries\b', re.IGNORECASE),
    'Brentford': re.compile(r'\bBrentford\b|\bBees\b', re.IGNORECASE),
    'Brighton': re.compile(r'\bBrighton\b|\bSeagulls\b', re.IGNORECASE),
    'Burnley': re.compile(r'\bBurnley\b|\bClarets\b', re.IGNORECASE),
    'Cardiif City': re.compile(r'\bCardiff\b|\bBluebirds\b', re.IGNORECASE),
    'Chelsea': re.compile(r'\bChelsea\b|\bBlues\b', re.IGNORECASE),
    'Crystal Palace': re.compile(r'\bCrystal Palace\b|\bEagles\b', re.IGNORECASE),
    'Everton': re.compile(r'\bEverton\b|\bToffees\b', re.IGNORECASE),
    'Fulham': re.compile(r'\bFulham\b|\bCottagers\b', re.IGNORECASE),
    'Hull City': re.compile(r'\bHull\b|\bTigers\b', re.IGNORECASE),
    'Huddersfield Town': re.compile(r'\bHuddersfield\b|\bTerriers\b', re.IGNORECASE),
    'Leeds United': re.compile(r'\bLeeds\b|\bLeeds United\b', re.IGNORECASE),
    'Leicester City': re.compile(r'\bLeicester\b|\bFoxes\b', re.IGNORECASE),
    'Liverpool': re.compile(r'\bLiverpool\b|\bReds\b', re.IGNORECASE),
    'Manchester City': re.compile(r'\bManchester City\b|\bCity\b', re.IGNORECASE),
    'Manchester United': re.compile(r'\bManchester United\b|\bUnited\b|\bRed Devils\b', re.IGNORECASE),
    'Middlesbrough': re.compile(r'\bMiddlesbrough\b|\bBoro\b', re.IGNORECASE),
    'Newcastle United': re.compile(r'\bNewcastle\b|\bMagpies\b', re.IGNORECASE),
    'Norwich City': re.compile(r'\bNorwich\b|\bCanaries\b', re.IGNORECASE),
    'Nottingham Forest': re.compile(r'\bNottingham Forest\b|\bForest\b', re.IGNORECASE),
    'Sheffield United': re.compile(r'\bSheffield United\b|\bBlades\b', re.IGNORECASE),
    'Southampton': re.compile(r'\bSouthampton\b|\bSaints\b', re.IGNORECASE),
    'Stoke City': re.compile(r'\bStoke\b|\bPotters\b', re.IGNORECASE),
    'Sunderland': re.compile(r'\bSunderland\b|\bBlack Cats\b', re.IGNORECASE),
    'Swansea City': re.compile(r'\bSwansea\b|\bSwans\b', re.IGNORECASE),
    'Tottenham Hotspur': re.compile(r'\bTottenham\b|\bSpurs\b', re.IGNORECASE),
    'Watford': re.compile(r'\bWatford\b|\bHornets\b', re.IGNORECASE),
    'West Brom': re.compile(r'\bWest Brom\b|\bBaggies\b', re.IGNORECASE),
    'West Ham United': re.compile(r'\bWest Ham\b|\bHammers\b', re.IGNORECASE),
    'Wolves': re.compile(r'\bWolves\b|\bWolverhampton\b', re.IGNORECASE)
}`

2. Initialise a dictionary that defaults all values to 0 so we can increment mention counts for each term easily

`mention_counts = defaultdict(int)`

3. Goes through each headline in our DataFrame, checking if each team appears in the headline using regex and if it does adds 1 to that team's mention count.

`for headline in df_2023['headline']:
    for team, pattern in patterns.items():
        if pattern.search(headline):
            mention_counts[team] += 1`

4. Converts the dictionary of mention counts into a new Data Frame with columns 'team' and 'mentions'

`df_mentions = pd.DataFrame(list(mention_counts.items()), columns=['team', 'mentions'])`

5. Sorts the teams from most mentioned to least mentioned
   
`df_mentions = df_mentions.sort_values(by='mentions', ascending=False)`

6. Extracts and sorts a list of all unique team names from the wins Data Frame

`all_teams = sorted(df_team_wins['team'].unique())`

7. Generates a unique colour for each team using a Seaborn colour palette and then stores the team colour pairs in a dictionary for consistent colour usage across plots

`team_colors = sns.color_palette("hsv", len(all_teams))
color_dict = {team: color for team, color in zip(all_teams, team_colors)}`

8. Sets a spacing value to slightly offset spacing labels

`offset = 3`

9. Creates a new figure for plotting the total wins, with a custom size

`plt.figure(figsize=(12, 8))`

10. Draws a smoothed black line across the data points

`sns.lineplot(x=df_team_wins['wins'], y=np.arange(len(df_team_wins)), ci=None, lw=2, color="black", estimator=None)`

11. It draws a horzontal line from 0 to each teams win count, then adds a circular marker at the end of the line and then annotates the marker with the win count slightly to the right

`for team, wins in zip(df_team_wins['team'], df_team_wins['wins']):
    plt.hlines(y=team, xmin=0, xmax=wins, color=color_dict[team], linewidth=2)
    plt.plot(wins, team, "o", color=color_dict[team], markersize=8)
    plt.text(wins + offset, team, str(wins), va='center', ha='left', fontsize=10, color='black')`

12. Adds x-axis label, plot title, gridlines, x-axis ticks, adjusts layout and finally displays the plot
     
`plt.xlabel("Total Wins ")
plt.title("Total Wins by Team (2015-2023)")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.xticks(np.arange(0, 230, 10))
plt.tight_layout()
plt.show()`

13. Starts a second figure

`plt.figure(figsize=(12, 10))`

14. Draws a smoothed line across mention points

`sns.lineplot(x=df_mentions['mentions'], y=np.arange(len(df_mentions)), lw=2, color="black", estimator=None, errorbar=None)`

15. Makes sure all teams in mentions have a corresponding colour if not grey as a fallback

`for team in df_mentions['team']:
    if team not in color_dict:
        color_dict[team] = 'grey'`

16. Similarly to our other plot, draws a horizontal line for each team and adds a marker and a label with the mentions count

`for team, mentions in zip(df_mentions['team'], df_mentions['mentions']):
    plt.hlines(y=team, xmin=0, xmax=mentions, color=color_dict[team], linewidth=2)
    plt.plot(mentions, team, "o", color=color_dict[team], markersize=8)
    plt.text(mentions + offset, team, str(mentions), va='center', ha='left', fontsize=10, color='black')`

17. Adds labels, title, grid and x-axis ticks for the mention count. Then does final layout cleaning and displays plot

`plt.xlabel("Number of Mentions in Headlines ")
plt.title("Premier League Team Mentions in Headlines (2013-2023) ", fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.xticks(np.arange(0, 190, 10))
plt.tight_layout()
plt.show()`

18. Creates a new figure and sets the figure size to 12x8

`plt.figure(figsize=(12, 8))`

19. Inner merge of both teams to ensure both teams are present

`df_combined = pd.merge(df_team_wins, df_mentions, on='team', how='inner')`

20. Loops through each team to plot its individual data point, mentions on the x-axis, wins on the y-axis, using smaller data points and using the colour dictioanry previously defined

`for team in df_combined['team']:
    plt.scatter(
        df_combined[df_combined['team'] == team]['mentions'],
        df_combined[df_combined['team'] == team]['wins'],
        color=color_dict[team],
        s=50 
    )`

21. Draws a dashed red regression line

`sns.regplot(
    x='mentions',
    y='wins',
    data=df_combined,
    scatter=False,
    color='red',  # Changed to red
    line_kws={'linestyle': '--', 'alpha': 0.7}
)`


22. Adds clear axis labels and a title, enabling a dashed grid, using tight_layout() to avoid overlaps and plt.show() to display the plot

`plt.xlabel("Number of Mentions in Headlines")
plt.ylabel("Total Wins (2015-2023)")
plt.title("Relationship Between Team Wins and Media Mentions")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()`

23. Calculates the Pearson correlation coefficient between our columns

`correlation = df_combined['wins'].corr(df_combined['mentions'])`

24. Prints the correlation value rounded to three decimal place

`print(f"\nCorrelation between wins and mentions: {correlation:.3f}")`

25. Judges the strength of our correlation

`if abs(correlation) > 0.7:
    strength = "strong"
elif abs(correlation) > 0.3:
    strength = "moderate"
else:
    strength = "weak"`

26. Checks the sign of the correlation

`direction = "positive" if correlation > 0 else "negative"`

27. Prints the result

`print(f"This indicates a {strength} {direction} correlation between team wins and media mentions.")`

# Import of GDP data [OUTPUT 6]


1. This calls the GDP.csv by locating it within my file directory skipping 4 rows as these are empty saving to df_gdp.

`df_gdp = pd.read_csv('/Users/georgewalsh/Desktop/API_NY/GDP.csv', skiprows=4)  # Skip the first 4 rows which contain metadata`

2. df_gdp_cleaned calls only the columns of interest.

`df_gdp_cleaned = df_gdp[['Country Name','2014','2015','2016','2017','2018','2019', '2020', '2021', '2022','2023']]`

3. gdp_columns is all the columns containing data of interest we then convert all these values to numeric

`gdp_columns = ['2014','2015','2016','2017','2018','2019', '2020', '2021', '2022', '2023']
df_gdp_cleaned[gdp_columns] = df_gdp_cleaned[gdp_columns].apply(pd.to_numeric, errors='coerce')  # Convert values`

4. We then create a new column total GDP which summarises the data in all our columns of interest.

`df_gdp_cleaned['Total GDP'] = df_gdp_cleaned[gdp_columns].sum(axis=1)`

5.Finally ,we test the success of our table manipulation by calling the first 5 lines of our data frame.

`df_gdp_cleaned.head()`

# Visualisation country mentions against total GDP [OUTPUT 7]


1. Imports given we have already imported most packages only need import lowess this function is used for stats models in our case it’s for a smoothest line of best fit.

`from statsmodels.nonparametric.smoothers_lowess import lowess`

2. Target countries specifies the list of countries we will be performing analysis on.

`target_countries = [
    'United States', 'China', 'Japan', 'Germany', 
    'India', 'United Kingdom', 'France', 'Italy',
    'Canada', 'Brazil', 'Russia', 'South Korea'
    ]`

3. Country variants ensure we account for variations of the words in our headline to minimise the number of articles discussing these countries that are missed.

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

4. This creates an empty dictionary to store our mentions. For each country and its variants, it will create a regex pattern to match whole words only and counts how many headlines contain each variant and sums across all variants in the country storing the total in a dictionary.

`country_mentions = {}
for country, variants in country_variants.items():
    total = 0
    for variant in variants:
        pattern = r'\b' + re.escape(variant) + r'\b'
        count = df['headline'].str.contains(pattern, case=False, regex=True).sum()
        total += count
    country_mentions[country] = total`

5. Creates a list of mention counts in the same order as target_countries.

`mentions_counts = [country_mentions[country] for country in target_countries]`

6. This filters the GDP dataframe to only include our countries of interest and also converts country names to a catagorical variable with our specified ordr and sorts the dataframe to match our target order.

`df_filtered = df_gdp_cleaned[df_gdp_cleaned['Country Name'].isin(target_countries)]
df_filtered['Country Name'] = pd.Categorical(
    df_filtered['Country Name'], 
    categories=target_countries,
    ordered=True
    )
df_filtered = df_filtered.sort_values('Country Name')`

7. Creates a figure containing both our subplots.

`plt.figure(figsize=(14, 10))`

8. For our first plot this creates the top plot making a bar chart of mention counts using distinct colours to aid the ability to distinguish between countries in my visualisation.

`plt.subplot(2, 1, 1)
bars = plt.bar(target_countries, mentions_counts, color=plt.cm.tab20.colors[:12])`

9. This adds count labels to each bar by iterating through each bar in the chart retrieving the value on the y axis and then adds text to the specific coordinate such that it finds the left position of the bar then adds half the width of the bar for central top of each bar.

`for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')`

10. This creates a smoothed red trend line through the bar heights.

`mentions_smoothed = lowess(mentions_counts, np.arange(len(target_countries)), frac=0.3)
plt.plot(target_countries, mentions_smoothed[:, 1], color='red', lw=2, label='Trend Line')`

11. This is general formatting such that it adds a title, rotates our x-ticks, adds grid lines and a legend. 

`plt.title('Country Mentions in Headlines (Top 12 Economies) (2014-2023)')
plt.ylabel('Number of Mentions')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.legend()`

12. This creates a second subplot in the bottom plot which creates a bar chart using the viridis colourmap.

`plt.subplot(2, 1, 2)
sns.barplot(x='Country Name', y='Total GDP', data=df_filtered, palette='viridis', order=target_countries)`

13. General formatting of our second plot

`plt.xlabel('Country')
plt.ylabel('Total GDP in Trillions(in USD)')
plt.title('Total GDP of Top 12 Economies (2013-2023)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)`

14. Final formatting that ensures proper spacing between subplots and displays the final visualisation

`plt.legend()
plt.tight_layout()
plt.show()`

# War datafarme import [OUTPUT 8]


1. Utilises the pandas function to read the CSV into the df_war data frame, skipping the first 4 rows of the file 

`df_war =pd.read_csv('/Users/georgewalsh/Documents/battle/battledata.csv',skiprows=4)`

2. This creates a new dataframe which only stores our columns of interest 

`df_war_cleaned = df_war[['Country Name','2014','2015','2016','2017','2018','2019', '2020', '2021', '2022','2023']]`

3. This creates a list of years from 2014 to 2023, storing as strings

`year_columns = [str(year) for year in range(2014, 2024)]`

4. This adds a new total column to df_war_cleaned by summing all the year columns

`df_war_cleaned['Total'] = df_war_cleaned[year_columns].sum(axis=1)`

5. This verifies it works by calling the first 5 rows

`df_war_cleaned.head()`

# Visualisation of total deaths by war against total war mentions [OUPUT 4]


1. This first defines our countries of interest

`key_countries = ['Ukraine', 'Russia', 'United States', 'Sudan', 'United Kingdom',
                 'Afghanistan', 'Ethiopia', 'Iraq']`

2. This accounts for all variations of the word 

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

5. This loops through each country and its variations

`for country, variations in country_variations.items():`

6. This creates a Boolean mask for headlines containing "war" or "conflict"

 `mask = df['headline_lower'].str.contains(r'\b(war|conflict)\b', case=False)`

7. This combines with another mask checking for any country variation in the text

 `country_mask = mask & df['headline_lower'].apply(
        lambda text: any(variant in text for variant in variations)`

8. This counts the matches and stores in a dictionary

 `mention_counts[country] = country_mask.sum()`

9. This converts the dictionary to a Data Frame with columns Country and Mentions

`mentions_df = pd.DataFrame(list(mention_counts.items()), columns=['Country', 'Mentions'])`

10. This filters the war deaths data to only include our key countries

`df_war_subset = df_war_cleaned[df_war_cleaned['Country Name'].isin(key_countries)]`

11. This will sort our data frame by total deaths in descending order

`df_war_subset = df_war_subset.sort_values('Total', ascending=False)`

12. This creates a 2-row, 1-column figure and sets the figure size to 12x12 inches

`fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))`

13. This plots the first total deaths plot in the colour sky blue separating by country

`bars1 = ax1.bar(df_war_subset['Country Name'], df_war_subset['Total'], color='skyblue')`

14.This format the figure adding a title, y-axis label and faint dotted grid lines on y-axis

`ax1.set_title('Total Deaths (2014–2023)', fontsize=14)
ax1.set_ylabel('Total Deaths')
ax1.grid(axis='y', linestyle=':', alpha=0.5)`

15. This adds formatted value labels to each bar using the same method as before but adding thousand separators

`for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', 
             ha='center', va='bottom', fontsize=10)`

16. This plots our second plot mentions reordering mentions data to match deaths plot order

`mentions_df = mentions_df[mentions_df['Country'].isin(key_countries)]
mentions_df = mentions_df.set_index('Country').loc[df_war_subset['Country Name']].reset_index()`

17. Prepare's data for smoothing by extracting x and y values

`x_vals = np.arange(len(mentions_df))
y_vals = mentions_df['Mentions'].values`

18. This creates a smooth trend line using spline interpolation only if there are enough data points plotted as a dash line

`if len(x_vals) > 2:
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    spline = make_interp_spline(x_vals, y_vals, k=2)  # you can adjust 'k' for curve smoothness
    y_smooth = spline(x_smooth)
    ax2.plot(x_smooth, y_smooth, color='darkred', linestyle='--', linewidth=2, label='Smoothed Trend')`

19. This creates bar plot of mentions in cornflower blue 

`bars2 = ax2.bar(mentions_df['Country'], mentions_df['Mentions'], color='cornflowerblue')`

20. This formats our graph adding titles, grid lines on y axis and rotating labels for readability 

`ax2.set_title('Mentions of "War" or "Conflict" in Headlines (2013-2021)', fontsize=14)
ax2.set_ylabel('Number of Mentions')
ax2.set_xticklabels(mentions_df['Country'], rotation=30)
ax2.grid(axis='y', linestyle=':', alpha=0.5)`

21. This adds value labels above bars similar to first plot

`for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height}', 
             ha='center', va='bottom', fontsize=10)`

22. Final formatting

`plt.tight_layout()
plt.show()`

# Political party data import [OUTPUT 9]

1.This reads the excel file indicating the headers are in row 9

`df_pol = pd.read_excel('/Users/georgewalsh/Documents/pivottablefull.xlsx', header=8)`

2. This filters the data to only include rows where 'Data' column equals 'Sum of Vote' and then only selecting the 'Party' column and the years '2015','2017' and '2019' then creating a new data frame with this subset of data 

`df_vote = df_pol[df_pol['Data'] == 'Sum of Vote'][['Party', 2015, 2017, 2019]]`

3. This creates a new column 'Total Votes' which calculates the sum of votes across the three election years in a row-wise summation

`df_vote['Total Votes'] = df_vote[[2015, 2017, 2019]].sum(axis=1)`

4.This calculates the grand total of all votes across parties

`total_all_parties = df_vote['Total Votes'].sum()`

5. Creates a new column 'Percentage' which calculates each party’s percentage by total vote 

`df_vote['Percentage'] = (df_vote['Total Votes'] / total_all_parties) * 100`

6. Replaces abbreviated party names with their full names using a dictionary mapping

`df_vote['Party'] = df_vote['Party'].replace({
    'CON': 'Conservative',
    'LAB': 'Labour',
    'LIB': 'Lib Dem',
    'NAT': 'Scotish National Party'
})`

7. Filters out using '~' logic (not) and isin to remove 'MIN' and 'OTH' parties

`df_vote = df_vote[~df_vote['Party'].isin(['MIN', 'OTH'])]`

8. These order the data frame by percentage column in order from largest to smallest percentage

`df_vote = df_vote.sort_values('Percentage', ascending=False)`

9. Matches the years of this data with our Guardian dataset

`df_2021= df[df['year'] <= 2021]`


# Visualisation of party votes against party mentions [OUTPUT 10]

1. Imports defaultdict which behaves like a normal dictionary but gives a default value(in this case, int, which is 0

`from collections import defaultdict`

2. Creates a dictionary called patterns where keys are party names, values are compiled regular expressions, our strings aren't case sensitive, and we don't match parts of words

`patterns = {
    'Conservative': re.compile(r'\bConservative(s)?\b|\bTory\b|\bTories\b', re.IGNORECASE),
    'Labour': re.compile(r'\bLabour\b', re.IGNORECASE),
    'Liberal Democrat': re.compile(r'\bLiberal Democrat(s)?\b|\bLib Dem(s)?\b', re.IGNORECASE),
    'Scottish National Party': re.compile(r'\bScottish National Party\b|\bSNP\b', re.IGNORECASE)
}`

3. Creates a dictionary with default integer values (0)

`mention_counts = defaultdict(int)`

4. Iterated over every headline in the Data Frame, using a set to avoid double counting, checking if any party's regex pattern is found in headline, if adding to found_parties and then finally each mentioned party has its count incremented by 1 in mention_counts

`for headline in df_2021['headline']:
    found_parties = set()
    for party, pattern in patterns.items():
        if pattern.search(headline):
            found_parties.add(party)
    for party in found_parties:
        mention_counts[party] += 1`

5. Converts the dictionary into a pandas Data Frame with each party becoming a row, and their counts stored in "Count" column

`count_df = pd.DataFrame.from_dict(mention_counts, orient='index', columns=['Count'])`

6. Calculates the total number of party mentions

`total_mentions = count_df['Count'].sum()`

7.Converts count to percentages of total mentions

`count_df['Percentage'] = (count_df['Count'] / total_mentions) * 100`

8. Sorts parties by percentage in descending order

`count_df = count_df.sort_values('Percentage', ascending=False)`

9. Sets consistent colour codes for the parties

`colors = {
    'Conservative': '#0087DC',
    'Labour': '#E4003B',
    'Liberal Democrat': '#FAA61A',
    'Scottish National Party': '#3F8428',
    'Green': '#6D3177',
    'Reform UK': '#999999'`

10. Starts a 8x8 plot

`plt.figure(figsize=(8, 8))`

11. Defines the colours to match the party order in the df_vote Data Frame

`colors_vote = ['#0087DC', '#E4003B', '#FAA61A', '#3F8428', '#6D3177', '#999999']`

12. Plots the vote share as a pie chart, showing percentages, starting the first slice at top and adding a white border for better separation

`plt.pie(df_vote['Percentage'], labels=df_vote['Party'], autopct='%1.1f%%',
        colors=colors_vote, startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})`

13. Adds a title and displays the chart

`plt.title('UK General Elections Total Vote Share by Party\n(2015, 2017 & 2019)', pad=20)
plt.tight_layout()
plt.show()`

14. Starts a new plot for second chart

`plt.figure(figsize=(8, 8))`

15. Fetches the colour for each party and defaulting to grey if party isn't in the colour dictionary

`colors_mentions = [colors.get(party, '#CCCCCC') for party in count_df.index]`

16. Plots a pie chart showing percentage of mentions and formatting the same as previous

`plt.pie(count_df['Percentage'], labels=count_df.index, autopct='%1.1f%%',
        colors=colors_mentions, startangle=90, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})`

17. Adds a title and displays the second chart

`plt.title('Political Party Mentions in Headlines\n(% of Total Mentions) (2013-2021)', pad=20)
plt.tight_layout()
plt.show()`














