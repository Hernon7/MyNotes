# Python cheat sheet for Exploratory Data Analysis (**EDA**)

## Table of contents

1. [Import pakages](#Importpackages)
2. [ Dataframe](#Dataframe)
    1. [Dataframe Properties](#DataframeProperties)
    2. [Missing Values](#MissingValues)
    3. [Numerical Values](#NumericalValues)
    4. [Categoerical Values](#CategoericalValues)
    5. [Time Values](#TimeValues)
3. [List](#ListProperties)
4. [Dictionary](#DictionaryProperties)
5. [String](#StringProperties)
6. [Stats Test](#StatsTest)
7. [Data Visualization](#DataVisualization)
8. [Functions](#UsefulFunctions)


## Import packages <a name="Importpackages"></a>

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import seaborn as sns
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```



## Dataframe

### Dataframe Properties <a name="DataframeProperties"></a>

#### Load Data

```python
#Load data from Excel tab
df = pd.read_excel (r'filename.xlsx', sheet_name='tab1',header = 1)
```



#### Basic Info

```python
df.info()
df.describe()
```
#### Get col and row number for a Dataframe
```python
count_row = df.shape[0]
count_col = df.shape[1]
```
#### Get appear number of different value type for categoric data

```python
s.value_counts(normalize=True)
```

#### Selecting

```python
anime['genre'].tolist()
anime['genre']
```

#### Get a list of index values

```python
anime_modified.index.tolist()
```

#### Get a list of column values

```python
anime.columns.tolist()
```

#### Append new column with a set value

```python
anime['train set'] = True
```

#### Create new data frame from a subset of columns

```python
anime[['name','episodes']]
```

#### Drop specified columns

```python
df.drop(columns=['col1','col1'],inplace = True)
anime.drop(['anime_id', 'genre', 'members'], axis=1).head()
```
#### Add a row with sum of other rows

We’ll manually create a small data frame here because it’s easier to look at. The interesting part here is `df.sum(axis=0)` which adds the values across rows. Alternatively `df.sum(axis=1)` adds values across columns.

The same logic applies when calculating counts or means, ie: `df.mean(axis=0)`.

```python
df = pd.DataFrame([[1,'Bob', 8000],
                  [2,'Sally', 9000],
                  [3,'Scott', 20]], columns=['id','name', 'power level'])df.append(df.sum(axis=0), ignore_index=True)
```

#### Concatenate 2 dataframes

```python
df1 = anime[0:2]
df2 = anime[2:4]
pd.concat([df1, df2], ignore_index=True)
```

#### Merge dataframes

This functions like a SQL left join, when you have 2 data frames and want to join on a column.

```python
rating.merge(anime, left_on=’anime_id’, right_on=’anime_id’, suffixes=(‘_left’, ‘_right’))
```

#### Retrieve rows with matching index values

```python
anime_modified.loc[['Haikyuu!! Second Season','Gintama']]
```

#### Retrieve rows by numbered index values

```python
anime_modified.iloc[0:3]
```

#### Get rows

Retrieve rows where a column’s value is in a given list. `anime[anime[‘type’] == 'TV']` also works when matching on a single value.

```python
anime[anime['type'].isin(['TV', 'Movie'])]
```

#### Slice a dataframe

```python
anime[1:3]
```

#### Filter by value

```python
anime[anime['rating'] > 8]	
```
#### Filter by list

```python
#in list
df[df['col'].isin(list)]
# not in list
df[~df['col'].isin(list)]
```

#### Filter columns that contain specific string

```python
quantity_col = [col for col in rawdf.columns if 'Quantity' in col]
```
#### sort_values

Sort data frame by values in a column.

```
anime.sort_values('rating', ascending=False)
```

#### Melt

#### Melt

**For dataframe like:**

| Items |  2020  |  2019  |  2018  |  2017  |  2016  |
| :---: | :----: | :----: | :----: | :----: | :----: |
| Item1 | Value1 | Value2 | Value3 | Value4 | Value5 |

**Transfer to:**

| Items | Year | Values |
| :---: | :--: | :----: |
| Item1 | 2020 | Value1 |
| Item1 | 2019 | Value2 |
| Item1 | 2018 | Value3 |
| Item1 | 2017 | Value4 |
| Item1 | 2016 | Value5 |

```python
df.melt(id_vars=["Items"], var_name="Year", value_name="Values")
```

#### Groupby in a Dataframe

```python
#Customize the display
groupby_A = df.groupby('A')
X = groupby_A.agg({'B': ['min', 'max'], 'C': 'sum'})
X.reset_index().pivot(index='A', columns='C', values='sum')
```

| Function   |             Description             |
| :--------- | :---------------------------------: |
| `count`    |   Number of non-null observations   |
| `sum`      |            Sum of values            |
| `mean`     |           Mean of values            |
| `mad`      |       Mean absolute deviation       |
| `median`   |     Arithmetic median of values     |
| `min`      |               Minimum               |
| `max`      |               Maximum               |
| `mode`     |                Mode                 |
| `abs`      |           Absolute Value            |
| `prod`     |          Product of values          |
| `std`      |     Unbiased standard deviation     |
| `var`      |          Unbiased variance          |
| `sem`      | Unbiased standard error of the mean |
| `skew`     |   Unbiased skewness (3rd moment)    |
| `kurt`     |   Unbiased kurtosis (4th moment)    |
| `quantile` |    Sample quantile (value at %)     |
| `cumsum`   |           Cumulative sum            |
| `cumprod`  |         Cumulative product          |
| `cummax`   |         Cumulative maximum          |
| `cummin`   |         Cumulative minimum          |
#### Counts

```python
#Find counts of each unique value in col
df.value_counts()
# Find counts
df.count()
```

#### Create a pivot table

```python
tmp_df = rating.copy()
tmp_df.sort_values('user_id', ascending=True, inplace=True)
tmp_df = tmp_df[tmp_df.user_id < 10] 
tmp_df = tmp_df[tmp_df.anime_id < 30]
tmp_df = tmp_df[tmp_df.rating != -1]
pd.pivot_table(tmp_df, values='rating', index=['user_id'], columns=['anime_id'], aggfunc=np.sum, fill_value=0)
```

#### Distinct value

```python
print('There are %d distinct value' %(df['col'].unique().shape))
```




#### Sample data

```python
anime.sample(frac=0.25)
```



### Missing Values <a name="MissingValues"></a>

#### Generate the boolean flags indicating missing rows and columns

```python
#find cols with missing values:
df.columns[df.isnull().sum(axis=0)>0]
crimes.columns[crimes.isnull().any()]
```



### Numerical Values <a name="NumericalValues"></a>

#### Remove the <$> in a Dataframe

```python
df['col'] = Cos_df['col'].apply(lambda x: float(x.replace('$','') if isinstance(x, str) else False))
```
#### Convert the percentage in to float in a Dataframe 

```python
df['col'].str.rstrip('%').astype('float') / 100.0
```



### Categoerical Values <a name="CategoericalValues"></a>

#### Cut a attribute into bins

```python
#Cut the column into 5 classes from 0 to 6
df["col"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```

```python
#use np.clip()
>>> a = np.arange(10)
>>> np.clip(a, 1, 8)
array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
```



### Time Values <a name="TimeValues"></a>

#### Time difference in days

```python
from datetime import timedelta, date
start_date = date(2016, 1, 27)
current_date = date(2020, 4, 14)
(current_date - start_date).days
```

#### Convert float to date

```python
timestamps = df['Time'].map(lambda t: str(int(t)) if not np.isnan(t) else '').map(lambda t:t if len(t)>2 else '')
df['Time2'] = timestamps.map(lambda t:'0'*(4-len(t))+t).map(lambda t:'%s:%s.000' %(t[:2], t[2:4]))
```
```python
def FuseDateTime(date, time):
    if type(date) == float or time == '': return np.nan
    return date + ' ' + time
df['Date Time'] = pd.to_datetime(list(map(FuseDateTime, crimes['Date'].values,  crimes['Time2'].values)))
```

#### Convert date to Year

```python
df['Year'] = df['Date Time'].map(lambda t: int(t.year) if t is not pd.NaT else None)
```

#### Convert date to Quarter

```python
df['Year_Quarter'] = df['Occurred Date Time'].map(lambda t: "%04dQ%d" %(t.year, (t.month-1)//3+1) if not np.isnan(t.year) else None)
```

#### Convert Quarter to Season

```python
quarterly_counts = df[(years>=2008)&(years<=2018)].groupby('Year_Quarter')['Report Number'].count()
quarters = quarterly_counts.index.str.replace('^\d{4}','')
ans = pd.DataFrame({'quarters':quarters, 'count':quarterly_counts.values}).groupby('quarters').mean()
ans.index = ['winter','spring','summer','autumn']
ans
```



---



## List Properties <a name="ListProperties"></a>

### Make a flat list out of list of lists

```python
flat_list = [item for sublist in l for item in sublist]
flatten = lambda l: [item for sublist in l for item in sublist]
```
---



## Dictionary Properties <a name="DictionaryProperties"></a>

### How to sort a dictionary according to values

```python
sorted(scores.items(), key = lambda x: x[1],reverse=True)[0:10]
```

>Dictionary in Python is an unordered collection of data values, used to store data values like a map, which unlike other Data Types that hold only single value as an element, Dictionary holds key : value pair.
>In Python Dictionary, items() method is used to return the list with all dictionary keys with values.

## String Properties <a name="StringProperties"></a>



---
## Stats Test <a name="StatsTest"></a>
>- The p-value is slightly below the  5%  threshold.
>- Based on  5%  condidence, we can reject the null hypothesis and accept the alternative.
  ANOVA
>- We may speculate that the X has played an essential role here.
>- If we test all features together, the p-value would be in-significant. This indicates that the additional features make the water muddy.
```python
from scipy.stats import f_oneway
```

---
## Data Visualization <a name="DataVisualization"></a>

### Save Charts

```python
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

### Historical Charts

```python
%matplotlib inline
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))
# save_fig("attribute_histogram_plots")
plt.show()
```

### Scatter Charts

```python
#1
df.plot(kind="scatter", x="x", y="y")
#2
df.plot(kind="scatter", x="x", y="y", alpha=0.1)
#3
housing.plot(kind="scatter", x="x", y="y", alpha=0.4,
    s=df["z"]/100, label="z", figsize=(10,7),
    c="c", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
```

### Correlation Charts

```python
corr_matrix = df.corr()
corr_matrix["col"].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```
---
## Useful Functions <a name="UsefulFunctions"></a>

### Function showing missing values

```python
def assess_NA(data):
    """
    Returns a pandas dataframe denoting the total number of NA values and the percentage of NA values in each column.
    The column names are noted on the index.
    Parameters
    ----------
    data: dataframe
    """
    # pandas series denoting features and the sum of their null values
    null_sum = data.isnull().sum()# instantiate columns for missing data
    total = null_sum.sort_values(ascending=False)
    percent = ( ((null_sum / len(data.index))*100).round(2) ).sort_values(ascending=False)
    # concatenate along the columns to create the complete dataframe
    df_NA = pd.concat([total, percent], axis=1, keys=['Number of NA', 'Percent NA'])
    # drop rows that don't have any missing data; omit if you want to keep all rows
    df_NA = df_NA[ (df_NA.T != 0).any() ]

    return df_NA
  
def missing_zero_values_table(df):
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " + str(mz_table.shape[0]) +
              " columns that have missing values.")
#         mz_table.to_excel('path/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
        return mz_table
```

### Function for unique values

```python
def assess_unique(data):
    df = data.value_counts().to_frame()
    count_row = df.shape[0]
    df = df.set_axis(['Counts'], axis=1, inplace=False)
    df['Percentage'] = df['Counts']/count_row
    return df
```

### Check NaN of a value

The usual way to test for a NaN is to see if it's equal to itself:

```python
def isNaN(num):
    return num != num
```
