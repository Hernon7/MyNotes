# Python cheat sheet for Exploratory Data Analysis (**EDA**)

## Table of contents

1. [Import pakages](#Import packages)
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


## Import packages

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

```
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

​```python
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

#### Sorted Value:

```python
df.sort_values(ascending=False).head(n)
```

#### Distinct value

```python
print('There are %d distinct value' %(df['col'].unique().shape))
```



#### Drop Columns from a Dataframe

```python
df.drop(columns=['col1','col1'],inplace = True)
```
#### Find columns that contain specific string

```python
quantity_col = [col for col in rawdf.columns if 'Quantity' in col]
```



### Missing Values <a name="MissingValues"></a>

#### Generate the boolean flags indicating missing rows and columns

```python
missingRows = pd.isnull(col).sum(axis=1) > 0
missingCols = pd.isnull(col).sum(axis=0) > 0
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
>Notice that we compare the winter vs summer quarters only. The p-value is slightly below the  5%  threshold.
>Based on  5%  condidence, we can reject the null hypothesis and accept the alternative.
ANOVA
We may speculate that the weather has played an essential role here.
If we test all four quarters together, the p-value would be in-significant. This indicates that the additional spring, autumn quarters make the water muddy.
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
