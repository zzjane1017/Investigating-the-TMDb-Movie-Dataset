

# Project: Investigating the TMDb Movie Dataset

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## 1. Introduction

> In this project, the TMDb movie data was selected to perform data analysis. This data set contains information about 10,000 movies collected from The Movie Database (TMDb), including user ratings and revenue. 

### 1.1 Attributes of dataset: 

* id
* imdb_id
* popularity
* budget
* revenue
* original_title
* cast
* homepage
* director
* tagline
* keywords
* overview
* runtime
* genres
* production_companies
* release_date
* vote_count
* vote_average
* release_year
* budget_adj
* revenue_adj

### 1.2 Questions about the dataset:
* Which genres are frequently occurred in movies?
* Which genres have higher popularity?
* Which genres make more profit? 
* Which genres have higher vote counts?
* Which movies have higher vote average over the year?



```python
# Import packages and load dateset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

% matplotlib inline
```

<a id='wrangling'></a>
## 2. Data Wrangling

> In this section of the report, data set will be loaded, at the same time, check for cleanliness, and then trim and clean the  dataset for analysis.
* Certain columns, like ‘cast’ and ‘genres’, contain multiple values separated by pipe (|) characters.
* There are some odd characters in the ‘cast’ column. Don’t worry about cleaning them. You can leave them as is.
* The final two columns ending with “_adj” show the budget and revenue of the associated movie in terms of 2010 dollars, accounting for inflation over time.

### 2.1 General Properties 


```python
# Load your data and print out a few lines
df = pd.read_csv('tmdb-movies.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>imdb_id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>cast</th>
      <th>homepage</th>
      <th>director</th>
      <th>tagline</th>
      <th>...</th>
      <th>overview</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135397</td>
      <td>tt0369610</td>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>
      <td>http://www.jurassicworld.com/</td>
      <td>Colin Trevorrow</td>
      <td>The park is open.</td>
      <td>...</td>
      <td>Twenty-two years after the events of Jurassic ...</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Universal Studios|Amblin Entertainment|Legenda...</td>
      <td>6/9/15</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76341</td>
      <td>tt1392190</td>
      <td>28.419936</td>
      <td>150000000</td>
      <td>378436354</td>
      <td>Mad Max: Fury Road</td>
      <td>Tom Hardy|Charlize Theron|Hugh Keays-Byrne|Nic...</td>
      <td>http://www.madmaxmovie.com/</td>
      <td>George Miller</td>
      <td>What a Lovely Day.</td>
      <td>...</td>
      <td>An apocalyptic story set in the furthest reach...</td>
      <td>120</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Village Roadshow Pictures|Kennedy Miller Produ...</td>
      <td>5/13/15</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>3.481613e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>262500</td>
      <td>tt2908446</td>
      <td>13.112507</td>
      <td>110000000</td>
      <td>295238201</td>
      <td>Insurgent</td>
      <td>Shailene Woodley|Theo James|Kate Winslet|Ansel...</td>
      <td>http://www.thedivergentseries.movie/#insurgent</td>
      <td>Robert Schwentke</td>
      <td>One Choice Can Destroy You</td>
      <td>...</td>
      <td>Beatrice Prior must confront her inner demons ...</td>
      <td>119</td>
      <td>Adventure|Science Fiction|Thriller</td>
      <td>Summit Entertainment|Mandeville Films|Red Wago...</td>
      <td>3/18/15</td>
      <td>2480</td>
      <td>6.3</td>
      <td>2015</td>
      <td>1.012000e+08</td>
      <td>2.716190e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>140607</td>
      <td>tt2488496</td>
      <td>11.173104</td>
      <td>200000000</td>
      <td>2068178225</td>
      <td>Star Wars: The Force Awakens</td>
      <td>Harrison Ford|Mark Hamill|Carrie Fisher|Adam D...</td>
      <td>http://www.starwars.com/films/star-wars-episod...</td>
      <td>J.J. Abrams</td>
      <td>Every generation has a story.</td>
      <td>...</td>
      <td>Thirty years after defeating the Galactic Empi...</td>
      <td>136</td>
      <td>Action|Adventure|Science Fiction|Fantasy</td>
      <td>Lucasfilm|Truenorth Productions|Bad Robot</td>
      <td>12/15/15</td>
      <td>5292</td>
      <td>7.5</td>
      <td>2015</td>
      <td>1.839999e+08</td>
      <td>1.902723e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>168259</td>
      <td>tt2820852</td>
      <td>9.335014</td>
      <td>190000000</td>
      <td>1506249360</td>
      <td>Furious 7</td>
      <td>Vin Diesel|Paul Walker|Jason Statham|Michelle ...</td>
      <td>http://www.furious7.com/</td>
      <td>James Wan</td>
      <td>Vengeance Hits Home</td>
      <td>...</td>
      <td>Deckard Shaw seeks revenge against Dominic Tor...</td>
      <td>137</td>
      <td>Action|Crime|Thriller</td>
      <td>Universal Pictures|Original Film|Media Rights ...</td>
      <td>4/1/15</td>
      <td>2947</td>
      <td>7.3</td>
      <td>2015</td>
      <td>1.747999e+08</td>
      <td>1.385749e+09</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



#### Assessing Data

* number of samples in dataset
* number of columns in dataset
* duplicate rows in dataset
* datatypes of columns
* features with missing values
* number of non-null unique values for features in dataset
* what those unique values are and counts for dataset


```python
# number of samples and columns in dataset
df.shape
```




    (10866, 21)




```python
# number of duplicated rows in dataset
df.duplicated().sum()
```




    1




```python
# datatypes of columns
df.dtypes
```




    id                        int64
    imdb_id                  object
    popularity              float64
    budget                    int64
    revenue                   int64
    original_title           object
    cast                     object
    homepage                 object
    director                 object
    tagline                  object
    keywords                 object
    overview                 object
    runtime                   int64
    genres                   object
    production_companies     object
    release_date             object
    vote_count                int64
    vote_average            float64
    release_year              int64
    budget_adj              float64
    revenue_adj             float64
    dtype: object




```python
# features of missing values
df.isnull().sum()
```




    id                         0
    imdb_id                   10
    popularity                 0
    budget                     0
    revenue                    0
    original_title             0
    cast                      76
    homepage                7930
    director                  44
    tagline                 2824
    keywords                1493
    overview                   4
    runtime                    0
    genres                    23
    production_companies    1030
    release_date               0
    vote_count                 0
    vote_average               0
    release_year               0
    budget_adj                 0
    revenue_adj                0
    dtype: int64




```python
# number of non-null unique values for features 
df.nunique()
```




    id                      10865
    imdb_id                 10855
    popularity              10814
    budget                    557
    revenue                  4702
    original_title          10571
    cast                    10719
    homepage                 2896
    director                 5067
    tagline                  7997
    keywords                 8804
    overview                10847
    runtime                   247
    genres                   2039
    production_companies     7445
    release_date             5909
    vote_count               1289
    vote_average               72
    release_year               56
    budget_adj               2614
    revenue_adj              4840
    dtype: int64



### 2.2 Data Cleaning 

* Drop extraneous columns.

  > * Features aren't relevant to popularity and profitability of the movie will be removed       
  > * profit will be used in this analysis 
  
* Drop any rows contain missing values.

* Drop any rows contain duplicated values.

* Split data for columns 'genres', which contains multiple values seperated by (|) 



```python
# Add in profit column 
df['profit'] = df['revenue'] - df['budget']
```


```python
# Drop extraneous columns
df.drop(['id','imdb_id', 'cast','director','keywords', 'production_companies',
        'homepage','tagline','overview','release_date', 'revenue_adj','budget_adj'], 
        axis=1, inplace=True)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>runtime</th>
      <th>genres</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1363528810</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.419936</td>
      <td>150000000</td>
      <td>378436354</td>
      <td>Mad Max: Fury Road</td>
      <td>120</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>228436354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.112507</td>
      <td>110000000</td>
      <td>295238201</td>
      <td>Insurgent</td>
      <td>119</td>
      <td>Adventure|Science Fiction|Thriller</td>
      <td>2480</td>
      <td>6.3</td>
      <td>2015</td>
      <td>185238201</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.173104</td>
      <td>200000000</td>
      <td>2068178225</td>
      <td>Star Wars: The Force Awakens</td>
      <td>136</td>
      <td>Action|Adventure|Science Fiction|Fantasy</td>
      <td>5292</td>
      <td>7.5</td>
      <td>2015</td>
      <td>1868178225</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.335014</td>
      <td>190000000</td>
      <td>1506249360</td>
      <td>Furious 7</td>
      <td>137</td>
      <td>Action|Crime|Thriller</td>
      <td>2947</td>
      <td>7.3</td>
      <td>2015</td>
      <td>1316249360</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View missing value count for each feature
df.isnull().sum()
```




    popularity         0
    budget             0
    revenue            0
    original_title     0
    runtime            0
    genres            23
    vote_count         0
    vote_average       0
    release_year       0
    profit             0
    dtype: int64




```python
# Drop rows with any null values in dataset
df.dropna(inplace=True)
```


```python
# Check if there is any null value
df.isnull().sum().any()
```




    False




```python
# Number of duplicates in dataset
df.duplicated().sum()
```




    1




```python
# Drop duplicates in dateset
df.drop_duplicates(inplace=True)
```


```python
# Check if there is any duplicated value
df.duplicated().sum()
```




    0




```python
# Save cleaned dataset
df.to_csv('movie_data_cleaned.csv', index=False)
```


```python
df = pd.read_csv('movie_data_cleaned.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>runtime</th>
      <th>genres</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1363528810</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.419936</td>
      <td>150000000</td>
      <td>378436354</td>
      <td>Mad Max: Fury Road</td>
      <td>120</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>228436354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.112507</td>
      <td>110000000</td>
      <td>295238201</td>
      <td>Insurgent</td>
      <td>119</td>
      <td>Adventure|Science Fiction|Thriller</td>
      <td>2480</td>
      <td>6.3</td>
      <td>2015</td>
      <td>185238201</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.173104</td>
      <td>200000000</td>
      <td>2068178225</td>
      <td>Star Wars: The Force Awakens</td>
      <td>136</td>
      <td>Action|Adventure|Science Fiction|Fantasy</td>
      <td>5292</td>
      <td>7.5</td>
      <td>2015</td>
      <td>1868178225</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.335014</td>
      <td>190000000</td>
      <td>1506249360</td>
      <td>Furious 7</td>
      <td>137</td>
      <td>Action|Crime|Thriller</td>
      <td>2947</td>
      <td>7.3</td>
      <td>2015</td>
      <td>1316249360</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Save splitted dataset
df.to_csv('movie_data_cleaned_splitted.csv', index=False)
```

<a id='eda'></a>
## Exploratory Data Analysis

> Distribution of variables and 
  Descriptive statistics will be applied in this section to explore the trend and relationship between features in dataset.



```python
df = pd.read_csv('movie_data_cleaned_splitted.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>runtime</th>
      <th>genres</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1363528810</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.419936</td>
      <td>150000000</td>
      <td>378436354</td>
      <td>Mad Max: Fury Road</td>
      <td>120</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>228436354</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.112507</td>
      <td>110000000</td>
      <td>295238201</td>
      <td>Insurgent</td>
      <td>119</td>
      <td>Adventure|Science Fiction|Thriller</td>
      <td>2480</td>
      <td>6.3</td>
      <td>2015</td>
      <td>185238201</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.173104</td>
      <td>200000000</td>
      <td>2068178225</td>
      <td>Star Wars: The Force Awakens</td>
      <td>136</td>
      <td>Action|Adventure|Science Fiction|Fantasy</td>
      <td>5292</td>
      <td>7.5</td>
      <td>2015</td>
      <td>1868178225</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.335014</td>
      <td>190000000</td>
      <td>1506249360</td>
      <td>Furious 7</td>
      <td>137</td>
      <td>Action|Crime|Thriller</td>
      <td>2947</td>
      <td>7.3</td>
      <td>2015</td>
      <td>1316249360</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Group the dateset by year
group_1 = df.groupby(['release_year']).mean()
group_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>profit</th>
    </tr>
    <tr>
      <th>release_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960</th>
      <td>0.458932</td>
      <td>6.892796e+05</td>
      <td>4.531406e+06</td>
      <td>110.656250</td>
      <td>77.531250</td>
      <td>6.325000</td>
      <td>3.842127e+06</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>0.422827</td>
      <td>1.488290e+06</td>
      <td>1.089420e+07</td>
      <td>119.419355</td>
      <td>77.580645</td>
      <td>6.374194</td>
      <td>9.405909e+06</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>0.454783</td>
      <td>1.710066e+06</td>
      <td>6.736870e+06</td>
      <td>124.343750</td>
      <td>74.750000</td>
      <td>6.343750</td>
      <td>5.026804e+06</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>0.502706</td>
      <td>2.156809e+06</td>
      <td>5.511911e+06</td>
      <td>111.323529</td>
      <td>82.823529</td>
      <td>6.329412</td>
      <td>3.355103e+06</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>0.412428</td>
      <td>9.400753e+05</td>
      <td>8.118614e+06</td>
      <td>109.214286</td>
      <td>74.690476</td>
      <td>6.211905</td>
      <td>7.178539e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the annual revenue, profit and budget of movies in each year
group_1[['revenue','profit','budget']].plot(
    title = 'Ecomonic perspective of movies over the year',
    color=('m','c','b'),linestyle=('-'),figsize=(10, 10));
```


![png](output_26_0.png)


The graph above shows that revenue, budget and profit are rapidly 
increasing from 1960 to 2000, but after year 2000, the development of
movie industry slows down especially in terms of profitability, which 
could be caused by ecomonic turbulance happened during that period of time.


```python
# Plot the trend of average vote score of movies in each year
group_1[['vote_average']].plot(title = 'Average vote of movies over the year',
        color=('g'),linestyle=('-'),figsize=(8, 8));
```


![png](output_28_0.png)


Supprisingly, the average vote score of movies reaches its peak at around
1972, but it starts to drop since then. Till 2010, average vote hits
its lowest value. This trend might explained by the increasing expectations from viewers. 


```python
# Plot the trend of average vote score of movies in each year
group_1[['vote_count']].plot(title = 'Features of movie data over the year',
           color=('k'),linestyle=('-'),figsize=(8, 8));
```


![png](output_30_0.png)


Even though the vote average score is decreasing steadily over the year, 
more votes received from viewers, on the contrary, keeps on upwards tendency,
it could caused by technology enhancement and diversity in viewers' profile.


```python
# Plot the trend of average vote score of movies in each year
group_1[['runtime']].plot(title = 'Duration of movies over the year',
           color=('y'),linestyle=('-'),figsize=(8, 8));
```


![png](output_32_0.png)


The duration of movies starts to decline since 1962, from highest `125min
to less than 100mins.


```python
# Group release year by countint the number of movies made each year
group_2 = df.groupby(['release_year']).count()
group_2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>runtime</th>
      <th>genres</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>profit</th>
    </tr>
    <tr>
      <th>release_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1960</th>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1961</th>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
    </tr>
    <tr>
      <th>1962</th>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1963</th>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1964</th>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the number of movies made by each year 
group_2[['original_title']].plot(title = 'Total number of movies produced over the year',
           color=('r'),linestyle=('-'),figsize=(8, 8));
```


![png](output_35_0.png)


From the graph above, it illustrates that from 1960 to 1980, the total number of movies produced are rising slowly, however, from 1980 to 2000, the growth of movie industry seems promising and after 2000, the number of movies surges to its peak value. 

### Research Question 1 (Which genres are frequently occurred in movies?)


```python
# Split multiple values in Genres column to extract more information about Genre
# Create a new datafreme with genres and counts
genre = df.genres.apply(lambda x: pd.value_counts(x.split('|'))).fillna(0)
genre.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Science Fiction</th>
      <th>Thriller</th>
      <th>Adventure</th>
      <th>Action</th>
      <th>Fantasy</th>
      <th>Crime</th>
      <th>Western</th>
      <th>Drama</th>
      <th>Family</th>
      <th>Animation</th>
      <th>Comedy</th>
      <th>Mystery</th>
      <th>Romance</th>
      <th>War</th>
      <th>History</th>
      <th>Music</th>
      <th>Horror</th>
      <th>Documentary</th>
      <th>TV Movie</th>
      <th>Foreign</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
genre_df = pd.DataFrame(genre.sum(axis=0), columns=['genre_count'])
genre_df['genres'] = genre_df.index
genre_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre_count</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Science Fiction</th>
      <td>1229.0</td>
      <td>Science Fiction</td>
    </tr>
    <tr>
      <th>Thriller</th>
      <td>2907.0</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>Adventure</th>
      <td>1471.0</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>Action</th>
      <td>2384.0</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>Fantasy</th>
      <td>916.0</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(style="whitegrid")
plt.subplots(figsize=(30,20))
# make barplot and sort bars
sns.barplot(y='genres', x='genre_count',data=genre_df,
            order=genre_df.sort_values('genre_count', 
            ascending=False).genres)
# Set labels
plt.xlabel("Genre counts", size=30)
plt.ylabel("Genres", size=30)
plt.title("Genres Counts in TMDb Movie Dataset", size=40);
```


![png](output_40_0.png)



```python
genre_df['genre_count'].plot.pie(title= 'Genres Counts in TMDb Movie Dataset (%)', figsize=(10,10), 
                                    autopct='%1.1f%%',fontsize=15);
```


![png](output_41_0.png)


Based on the bar plot and pie chart above, we can firstly tell that Drama, Comedy and Triller are most frequently occured genres in TMDb Movie Dataset, which followed by Action, Romance and Horror.

### Research Question 2  ( Which genres have higher popularity? )


```python
# Get accumulated popularity score for each genres 
genre_pop_df = pd.DataFrame((np.matrix(df.popularity) * np.matrix(genre)).T, columns=['genre_pop'])
genre_pop_df['genres'] = genre_df.index
genre_pop_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre_pop</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1230.902062</td>
      <td>Science Fiction</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2155.723620</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1697.915054</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2208.238255</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>4</th>
      <td>909.441171</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the relationship between genre and popularity in descending order
sns.set(style="whitegrid")
plt.subplots(figsize=(30,20))
# make barplot and sort bars
sns.barplot(y='genres', x='genre_pop',data=genre_pop_df,
            order=genre_pop_df.sort_values('genre_pop', 
            ascending=False).genres)
# Set labels
plt.xlabel("Popularity", size=30)
plt.ylabel("Genres", size=30)
plt.title("Accumulated popularity of each genre in TMDb Movie Dataset", size=40);
```


![png](output_45_0.png)


In terms of popularity, Drama, Comedy and Action movies are leading in chart, and followed by Triller, Adventure and Science Fiction. 

### Research Question 3  ( Which genres make more profit? )


```python
# Get accumulated popularity score for each genres 
genre_pro_df = pd.DataFrame((np.matrix(df.profit) * np.matrix(genre)).T, columns=['genre_pro'])
genre_pro_df['genres'] = genre_df.index
genre_pro_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre_pro</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.551132e+10</td>
      <td>Science Fiction</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.128174e+10</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.111990e+11</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.074395e+11</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.835018e+10</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the relationship between genre and profit in descending order
sns.set(style="whitegrid")
plt.subplots(figsize=(30,20))
# make barplot and sort bars
sns.barplot(y='genres', x='genre_pro',data=genre_pro_df,
            order=genre_pro_df.sort_values('genre_pro', 
            ascending=False).genres)
# Set labels
plt.xlabel("Profit", size=30)
plt.ylabel("Genres", size=30)
plt.title("Accumulated profit of each genre in TMDb Movie Dataset", size=40);
```


![png](output_49_0.png)


Result from this chart states that Advanture, Action and Comedy produce relatively more profit comparing with other genres.

### Research Question 4  ( Which genres have higher vote counts?  )


```python
# Get accumulated popularity score for each genres 
genre_vote_df = pd.DataFrame((np.matrix(df.vote_count) * np.matrix(genre)).T, columns=['genre_vote'])
genre_vote_df['genres'] = genre_df.index
genre_vote_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>genre_vote</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>537191.0</td>
      <td>Science Fiction</td>
    </tr>
    <tr>
      <th>1</th>
      <td>742693.0</td>
      <td>Thriller</td>
    </tr>
    <tr>
      <th>2</th>
      <td>754807.0</td>
      <td>Adventure</td>
    </tr>
    <tr>
      <th>3</th>
      <td>936897.0</td>
      <td>Action</td>
    </tr>
    <tr>
      <th>4</th>
      <td>385399.0</td>
      <td>Fantasy</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the relationship between genre and vote counts in descending order
sns.set(style="whitegrid")
plt.subplots(figsize=(30,20))
# make barplot and sort bars
sns.barplot(y='genres', x='genre_vote',data=genre_vote_df,
            order=genre_vote_df.sort_values('genre_vote', 
            ascending=False).genres)
# Set labels
plt.xlabel("Vote counts", size=30)
plt.ylabel("Genres", size=30)
plt.title("Accumulated vote count of each genre in TMDb Movie Dataset", size=40);
```


![png](output_53_0.png)


Action, Drama and Adventure movies received more vote counts from views, which follows by Thriller, Comedy and Science Ficture.

Meanwhile, some analysis on vote average and profitability of the movies has been done, it shows those movies with above median vote score of 6 perform well in profitability compares with movies with less than median vote score. 


```python
df['vote_average'].median()
```




    6.0




```python
# select samples with vote less than the median and get mean
low_vote = df.query('vote_average < 6').mean().profit

# select samples with vote greater than or equal to the median and get mean
high_vote = df.query('vote_average >= 6').mean().profit
```


```python
# Create a bar chart with proper labels
data = [low_vote, high_vote]
labels = ['Low', 'High']
sns.barplot(labels, data, tick_label=labels)
plt.title('Average profit produced by vote score')
plt.xlabel('Vote score')
plt.ylabel('Profit');
```


![png](output_57_0.png)


### Research Question 5  ( Which movies have higher vote average? )


```python
movie_pop = pd.DataFrame(df.groupby(['release_year','original_title'])['vote_average'].mean())
movie_pop.sort_values(by='vote_average',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>vote_average</th>
    </tr>
    <tr>
      <th>release_year</th>
      <th>original_title</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011</th>
      <th>The Story of Film: An Odyssey</th>
      <td>9.2</td>
    </tr>
    <tr>
      <th>2015</th>
      <th>The Mask You Live In</th>
      <td>8.9</td>
    </tr>
    <tr>
      <th>2010</th>
      <th>Life Cycles</th>
      <td>8.8</td>
    </tr>
    <tr>
      <th>2014</th>
      <th>Black Mirror: White Christmas</th>
      <td>8.8</td>
    </tr>
    <tr>
      <th>2006</th>
      <th>Pink Floyd: Pulse</th>
      <td>8.7</td>
    </tr>
    <tr>
      <th>2010</th>
      <th>Opeth: In Live Concert At The Royal Albert Hall</th>
      <td>8.6</td>
    </tr>
    <tr>
      <th>2008</th>
      <th>John Mayer: Where the Light Is Live in Los Angeles</th>
      <td>8.5</td>
    </tr>
    <tr>
      <th>1981</th>
      <th>Queen - Rock Montreal</th>
      <td>8.5</td>
    </tr>
    <tr>
      <th>1995</th>
      <th>A Personal Journey with Martin Scorsese Through American Movies</th>
      <td>8.5</td>
    </tr>
    <tr>
      <th>2000</th>
      <th>Dave Chappelle: Killin' Them Softly</th>
      <td>8.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot heatmap between vote average
movie_pop_pivot = pd.pivot_table(movie_pop, values='vote_average', index=['original_title'], columns=['release_year'])
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(movie_pop_pivot, cmap='YlGnBu');
```


![png](output_60_0.png)


It can be noticed from the list generated above, most of the high average vote score movies are produced after year 2000, and some of the movies are TV series, which could explain why they are made after year 2000. 

<a id='conclusions'></a>
## Conclusions

> **Limitations**: 
The hypotheses should be performed to further investigate the correlation and causal relationship between variables.


> **Findings**: 

The genre of movies should be selected if we are considering making a new movie. 

* Frequently occured: Drama, Comedy and Triller
* Popularity: Drama, Comedy and Action
* Profitability: Advanture, Action and Comedy
* Vote count: Action, Drama and Adventure

At the same time, we can watch the list of higher vote average movies as the reference before embarking on movie producing.



## Submitting your Project 

> Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).

> Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.

> Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!


```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
```




    0




```python

```
