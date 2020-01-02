# Python cheat sheet for EDA process
---
**import packages**

```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import seaborn as sns
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
```
---
>**get col and row for a dataframe**\
```python
count_row = df.shape[0]
count_col = df.shape[1]
```
>**get appear number of different value type for categoric data**\
`s.value_counts(normalize=True)`

>**generate the boolean flags indicating missing rows and columns**\
`missingRows = pd.isnull(ncbirths).sum(axis=1) > 0`\
`missingCols = pd.isnull(ncbirths).sum(axis=0) > 0`


>**columns sum(sum all rows)**\
`df.sum(axis=0)`

>**rows sum(sum all cols):**\
`df.sum(axis=1)`

>**how to sort a dictionary according to values**\
`sorted(scores.items(), key = lambda x: x[1],reverse=True)[0:10]`\
*Dictionary in Python is an unordered collection of data values, used to store data values like a map, which unlike other Data Types that hold only single value as an element, Dictionary holds key : value pair.*\
*In Python Dictionary, items() method is used to return the list with all dictionary keys with values.*

>**find columns that contain specific string**\
`quantity_col = [col for col in rawdf.columns if 'Quantity' in col]`
---
>**function used to deal with missing values**
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
---
>**function used to deal with missing values**
```
def assess_unique(data):
    df = data.value_counts().to_frame()
    count_row = df.shape[0]
    df = df.set_axis(['Counts'], axis=1, inplace=False)
    df['Percentage'] = df['Counts']/count_row
    return df
```
---
# Python cheat sheet for Machine Learning
---

