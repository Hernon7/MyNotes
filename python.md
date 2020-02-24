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

#### Function showing missing values

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

### Sklearn Functions for feature engineering

#### - Missing Values

##### SimpleImputer

```python
# Using SimpleImputer to fill missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(housing_num)
```

#### - Categorical varible

##### OrdinalEncoder

```python
# Using OrdinalEncoder to convert categorical data
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

ordinal_encoder.categories_
```

##### OneHotEncoder

```python 
# Using OneHotEncoder to create dummy varables
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```

##### LabelEncoder

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

#### - Feature Scaling

##### Normalization 

> Values are shifed and rescaled so that they end up ranging from 0 to 1

```python
from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()

```

##### Standardization 

> It substracts the mean value(so the standardized value always have a zero mean), and then it divids by the standard deviation so that the resulting distribution has unit variance. Unlike Nornalization, standardization does not bound values to a specific range.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
```

### Select and Train a Model

#### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

```

### Hyperparameter Tunning

#### Grid Search

```python

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)


grid_search.best_params_


grid_search.best_estimator_

# The score of each hyperparameter combination tested during the grid search:
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

#### Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

#### Feature Importance

```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```

#### Final Model

```python
final_model = grid_search.best_estimator_s
```

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

## Python cheat sheet for Deep Learning

---

### Neural Network Demo via Pytorch

```python

# Import the libraries we need for this lab
import torch 
import torch.nn as nn
from torch import sigmoid
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)

# The function for plotting the model

def PlotStuff(X, Y, model, epoch, leg=True):
    
    plt.plot(X.numpy(), model(X).detach().numpy(), label=('epoch ' + str(epoch)))
    plt.plot(X.numpy(), Y.numpy(), 'r')
    plt.xlabel('x')
    if leg == True:
        plt.legend()
    else:
        pass

# Define the class Net

class Net(nn.Module):
    
    # Constructor
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        # hidden layer 
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        # Define the first linear layer as an attribute, this is not good practice
        self.a1 = None
        self.l1 = None
        self.l2=None
    
    # Prediction
    def forward(self, x):
        self.l1 = self.linear1(x)
        self.a1 = sigmoid(self.l1)
        self.l2=self.linear2(self.a1)
        yhat = sigmoid(self.linear2(self.a1))
        return yhat

# Define the training function

def train(Y, X, model, optimizer, criterion, epochs=1000):
    cost = []
    total=0
    for epoch in range(epochs):
        total=0
        for y, x in zip(Y, X): #zip function return y,x which have same index
            yhat = model(x) #get the predict value
            loss = criterion(yhat, y) #get the value of loss function
            loss.backward() #calculate the gradient of loss function
            optimizer.step() #parameters update based on the gradient value stored in grad()
            optimizer.zero_grad() #In Pytorch, the default setting is sum the grad(), so have to set grad() to zero before the next loop
            #cumulative loss 
            total+=loss.item() 
        cost.append(total)
        if epoch % 300 == 0:   #for each 300 loops, draw a graph 
            PlotStuff(X, Y, model, epoch, leg=True)
            plt.show()
            model(X)
            plt.scatter(model.a1.detach().numpy()[:, 0], model.a1.detach().numpy()[:, 1], c=Y.numpy().reshape(-1))
            plt.title('activations')
            plt.show()
    return cost

# Make some data

X = torch.arange(-20, 20, 1).view(-1, 1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:, 0] > -4) & (X[:, 0] < 4)] = 1.0

# The loss function

def criterion_cross(outputs, labels):
    out = -1 * torch.mean(labels * torch.log(outputs) + (1 - labels) * torch.log(1 - outputs))
    return out

# Train the model
# size of input 
D_in = 1
# size of hidden layer 
H = 2
# number of outputs 
D_out = 1
# learning rate 
learning_rate = 0.1
# create the model 
model = Net(D_in, H, D_out)
#optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#train the model usein
cost_cross = train(Y, X, model, optimizer, criterion_cross, epochs=1000)
#plot the loss
plt.plot(cost_cross)
plt.xlabel('epoch')
plt.title('cross entropy loss')

#Train the model with MSE Loss Function

learning_rate = 0.1
criterion_mse=nn.MSELoss()
model=Net(D_in,H,D_out)
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
cost_mse=train(Y,X,model,optimizer,criterion_mse,epochs=1000)
plt.plot(cost_mse)
plt.xlabel('epoch')
plt.title('MSE loss ')
```

---

#### [Markdown Demo](https://markdown-it.github.io/)