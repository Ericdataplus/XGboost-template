
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor # I here just in case we want to call it
from sklearn.ensemble import RandomForestRegressor # I here just in case we want to call it
from sklearn import preprocessing 
from sklearn.metrics import mean_absolute_error # cross validation is better 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor # The best model for dataset tackling and to know


import missingno
%matplotlib inline
# Ignore warnings for jupyter notebooks
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn()     # This is the first cell to import everything in and setup.
# missingno.matrix(df, figsize = (30, 10)) # quick way to check for missing data

df = pd.read_csv('', encoding='UTF-8' or 'Latin1')
```

## This cell will be the steps with the kaggle links to quickly implement in own project quickest this is an attempt at complete effiency 

to the part that's a bit more complex 

- imputing and onehotencoding - [imputing](https://www.kaggle.com/alexisbcook/missing-values) and [onehotencoding](https://www.kaggle.com/alexisbcook/categorical-variables) but these two are bundled together in [pipelines kaggle course](https://www.kaggle.com/alexisbcook/pipelines) 

## then after that you can do the model running part and error evualtion part

- usually xgboost and mean absolute error with maybe some cross validation [XGBoost](https://www.kaggle.com/alexisbcook/xgboost) -  [cross validation](https://www.kaggle.com/alexisbcook/cross-validation)

As far as I know now that's the hardest part and then repeat through messing with training data and model parameters or even setting up different models to predict with their parameters. 

If ready for people to use as app or website think about data leakage [kaggle data leakage](https://www.kaggle.com/alexisbcook/data-leakage)

This cell here pretty much condensed all of this notebook lol

Above is pretty much all of what you need to know to implement machine learning in the real world when your new to the machine learning practice. 


```python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=6)

y = df['']
features = ['', '', '', '', '' , '', '', '']
X = df[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0)


my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_val, y_val)], 
             verbose=False)



predictions = my_model.predict(X_val)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_val)))
```

## Steps to consider

- [Pipelines](https://www.kaggle.com/alexisbcook/pipelines) - this makes missing values, categorical data transforming, choosing models, fitting the data and cross validation all simpler.
- [Missing Values](https://www.kaggle.com/alexisbcook/missing-values)
- [Categorical data](https://www.kaggle.com/alexisbcook/categorical-variables)
- [XGboost](https://www.kaggle.com/alexisbcook/xgboost)
- [Cross Validation](https://www.kaggle.com/alexisbcook/cross-validation)
- [Date Leakge](https://www.kaggle.com/alexisbcook/data-leakage)



```python

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# This step is isolating the data needed to be transformed usually with a list comprehension. 

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)

```


```python
# kaggle reference (https://www.kaggle.com/alexisbcook/pipelines)  

# pipeline 
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),   # pick model and imputer and onehotencoding
                              ('model', model)
                             ]) # usually imputer, onehotencoding and then model

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)  

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_val)

# Evaluate the model
score = mean_absolute_error(y_val, preds)
print('MAE:', score)

# pipelines make cross validation easier and the general ML workflow
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

print(scores.mean())
```


```python
# transforming cat to nums though pipeline should replace this
# relies on 
# from sklearn import preprocessing
# use this if I still don't get the pipeline way

le = preprocessing.LabelEncoder()
le.fit(df.fruit)
df['categorical_label'] = le.transform(df.fruit)

# Transform labels back to original encoding.

df['categorical_label'] = le.inverse_transform(df['categorical_label']) 
```


```python

```
