# Python cheat sheet for Exploratory Data Analysis (**EDA**)

## Useful Quarries

### Import packages

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

### Get col and row number for a dataframe

```python
count_row = df.shape[0]
count_col = df.shape[1]
```

### Get appear number of different value type for categoric data

```python
s.value_counts(normalize=True)
```

### Generate the boolean flags indicating missing rows and columns

```python
missingRows = pd.isnull(ncbirths).sum(axis=1) > 0
missingCols = pd.isnull(ncbirths).sum(axis=0) > 0
```

### Columns sum(sum all rows)

```python
df.sum(axis=0)
```

**Rows sum(sum all cols):**

```python
df.sum(axis=1)
```
### Make a flat list out of list of lists

```python
flat_list = [item for sublist in l for item in sublist]
flatten = lambda l: [item for sublist in l for item in sublist]
```

### How to sort a dictionary according to values

```python
sorted(scores.items(), key = lambda x: x[1],reverse=True)[0:10]
```

>Dictionary in Python is an unordered collection of data values, used to store data values like a map, which unlike other Data Types that hold only single value as an element, Dictionary holds key : value pair.\
>In Python Dictionary, items() method is used to return the list with all dictionary keys with values.

### Find columns that contain specific string

```python
quantity_col = [col for col in rawdf.columns if 'Quantity' in col]
```

---

## Useful Functions

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