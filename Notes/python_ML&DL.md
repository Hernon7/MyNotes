# Python cheat sheet for Machine Learning and Deep Learning

## Python cheat sheet for Machine Learning

### Sklearn Functions for feature engineering

#### Missing Values

```python
#Generate dataframe with missing values
sample_incomplete_rows = df[df.isnull().any(axis=1)].head()
```

```python
# 1.Get rid of the corresponding districts
df.dropna(subset=["total_bedrooms"])
```

```python
# 2.Get rid of the whole attribute
df.drop("col", axis=1) 
```

```python
# 3.Set the values to some value(zero,the mean,the median.etc.)
median = df["col"].median()
sample_incomplete_rows["col"].fillna(median, inplace=True)
```

```python
# Using SimpleImputer to fill missing values
from sklearn.impute import SimpleImputer
# Remove the text attribute because median can only be calculated on numerical attributes:
df_num = df.drop("text", axis=1)
def fill_missing(df,strategy='median'):
    imputer = SimpleImputer(strategy = strategy)
    imputer.fit(df)
    X = imputer.transform(df)
    df_tr = pd.DataFrame(X, columns=df.columns,index=df.index)
    print(imputer.statistics_)
    return df_tr
```

```python
# insert categorical missing value based on distribution
def fillingOnDistribute(df_missing):
    dis = df_missing.value_counts(normalize=True)
    missing = df_missing.isnull()
    df_missing.loc[missing] = np.random.choice(dis.index, size=sum(missing),p=dis.values)
    return df_missing
```



#### Categorical varible

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
df_cat_1hot = cat_encoder.fit_transform(df_cat)
housdf_cat_1hoting_cat_1hot.toarray()
```

```python
#Alternatively, you can set sparse=False when creating the OneHotEncoder:
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder(sparse=False)
df_cat_1hot = cat_encoder.fit_transform(df_cat)

cat_encoder.categories_
cat_tr = pd.DataFrame(df_cat_1hot, columns= cat_encoder.categories_,index=df_cat.index)
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

#### Feature Scaling

##### Normalization 

> Values are shifed and rescaled so that they end up ranging from 0 to 1

```python
from sklearn.preprocessing import MinMaxScaler
def normalize(df):
    scaler = MinMaxScaler()
    scaler.fit(df)
    scaler.transform(df)
    X = scaler.transform(df)
    df_normalized = pd.DataFrame(X, columns=df.columns,index=df.index)
    print(scaler.data_max_)
    return df_normalized
```

##### Standardization 

> It substracts the mean value(so the standardized value always have a zero mean), and then it divids by the standard deviation so that the resulting distribution has unit variance. Unlike Nornalization, standardization does not bound values to a specific range.
```python
from sklearn.preprocessing import StandardScaler
def standardize(df):
    scaler = StandardScaler()
    scaler.fit(df)
    scaler.transform(df)
    X = scaler.transform(df)
    df_standardized = pd.DataFrame(X, columns=df.columns,index=df.index)
    print(scaler.mean_)
    return df_standardized
```
##### Boxcox for feature skewness

```python
#boxcox()
"""
y = (x**lmbda - 1) / lmbda,  for lmbda > 0
    log(x),                  for lmbda = 0            
"""
from scipy import stats
stats.boxcox(x)
```

```python
#boxcox1p
"""
y = ((1+x)**lmbda - 1) / lmbda  if lmbda != 0
    log(1+x)                    if lmbda == 0
"""
from scipy.special import boxcox1p
boxcox1p(x, 0.25)
```

#### Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```

```python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
```



### Select and Train a Model

#### Training and Evaluating on the Training Set

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
model = LinearRegression()
model.fit(X, y)
```

```python
from sklearn.metrics import mean_squared_error
predictions = model.predict(housing_prepared)
mse = mean_squared_error(Y_test, predictions)
rmse = np.sqrt(mse)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test, predictions)
```

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
pd.Series(np.sqrt(-scores)).describe()
```

### Hyperparameter Tunning

#### Grid Search

```python
from sklearn.ensemble import RandomForestRegressor

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
grid_search.fit(X, y)
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
rnd_search.fit(X, y)

cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

#### Example SciPy distributions for RandomizedSearchCV

```python
from scipy.stats import geom, expon
geom_distrib=geom(0.5).rvs(10000, random_state=42)
expon_distrib=expon(scale=1).rvs(10000, random_state=42)
plt.hist(geom_distrib, bins=50)
plt.show()
plt.hist(expon_distrib, bins=50)
plt.show()
```

#### Feature Importance

```python
feature_importances = grid_search.best_estimator_.feature_importances_
attributes = list(X.columns)
sorted(zip(feature_importances, attributes), reverse=True)
```

#### Final Model

```python
final_model = grid_search.best_estimator_s
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