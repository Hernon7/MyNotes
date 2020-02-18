# <center>Python cheat sheet for Data Scienct</center>

## EDA Process

### Useful Quarries

#### Import packages

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

#### Get col and row number for a dataframe

```python
count_row = df.shape[0]
count_col = df.shape[1]
```

#### Get appear number of different value type for categoric data

```python
s.value_counts(normalize=True)
```

#### Generate the boolean flags indicating missing rows and columns

```python
missingRows = pd.isnull(ncbirths).sum(axis=1) > 0
missingCols = pd.isnull(ncbirths).sum(axis=0) > 0
```

#### Columns sum(sum all rows)

```python
df.sum(axis=0)
```

**Rows sum(sum all cols):**

```python
df.sum(axis=1)
```

#### How to sort a dictionary according to values

```python
sorted(scores.items(), key = lambda x: x[1],reverse=True)[0:10]
```

>Dictionary in Python is an unordered collection of data values, used to store data values like a map, which unlike other Data Types that hold only single value as an element, Dictionary holds key : value pair.\
In Python Dictionary, items() method is used to return the list with all dictionary keys with values.

#### Find columns that contain specific string

```python
quantity_col = [col for col in rawdf.columns if 'Quantity' in col]
```

---

### Useful Functions

#### Function for dealing with missing values

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

#### Function for dealing with unique values

```python
def assess_unique(data):
    df = data.value_counts().to_frame()
    count_row = df.shape[0]
    df = df.set_axis(['Counts'], axis=1, inplace=False)
    df['Percentage'] = df['Counts']/count_row
    return df
```

#### Check NaN of a value

The usual way to test for a NaN is to see if it's equal to itself:

```python
def isNaN(num):
    return num != num
```

---

## Python cheat sheet for Machine Learning

---

### RandomForest

>[RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)\
[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

#### Train & Fit

```python
#import ML packages
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators= 500, max_leaf_nodes= 16,n_jobs= -1)
rnd_clf.fit(X_train,Y_train)
```

#### LabelEncoder

```python
from sklearn import preprocessing 
#LabelEncoder: turn tring into incremental value
def LabelEncoder(df):
    le = preprocessing.LabelEncoder()
    for column_name in df.columns:
        if df[column_name].dtype == object:
            df[column_name] = le.fit_transform(df[column_name])
        else:
            pass
```

---

## PySpark

### Import packagess

```python
import findspark
import pyspark
findspark.init()
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
sc =SparkContext()#SparkContext
spark = SparkSession(sc)
```

### Create spark dataframe from a file

```python
# create a dataframe 
df = spark.read.parquet('file_path')

# register a corresponding query table
df.createOrReplaceTempView('df')
```

### Output dataframe and the Schema

```python
df.show()
df.printSchema()
```

### Using SQL in PySpark

```python
spark.sql('select class,count(*) from df group by class').show()
```

### Using PySpark function

```python
df.groupBy('class').count().show()
```

### Using pixiedust

```python
import pixiedust
from pyspark.sql.functions import col
counts = df.groupBy('class').count().orderBy('count')
display(counts)
```

### ETL process

```python
#import packages for ETL process
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

indexer = StringIndexer(inputCol="class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol="classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)
minmaxscaler = MinMaxScaler(inputCol="features_norm", outputCol="scaledFeatures")
pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer,minmaxscaler])
model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()
```

---

#### [Markdown Demo](https://markdown-it.github.io/)