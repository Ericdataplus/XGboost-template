# XGboost-template
This is starter code for a machine learning workflow using the best machine learning model for tabular data

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
```


```python
df = pd.read_csv('', encoding='UTF-8' or 'Latin1')
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-2-6281c11387ea> in <module>
    ----> 1 data = pd.read_csv('', encoding='UTF-8' or 'Latin1')
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in parser_f(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)
        674         )
        675 
    --> 676         return _read(filepath_or_buffer, kwds)
        677 
        678     parser_f.__name__ = name
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in _read(filepath_or_buffer, kwds)
        446 
        447     # Create the parser.
    --> 448     parser = TextFileReader(fp_or_buf, **kwds)
        449 
        450     if chunksize or iterator:
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in __init__(self, f, engine, **kwds)
        878             self.options["has_index_names"] = kwds["has_index_names"]
        879 
    --> 880         self._make_engine(self.engine)
        881 
        882     def close(self):
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in _make_engine(self, engine)
       1112     def _make_engine(self, engine="c"):
       1113         if engine == "c":
    -> 1114             self._engine = CParserWrapper(self.f, **self.options)
       1115         else:
       1116             if engine == "python":
    

    ~\anaconda3\lib\site-packages\pandas\io\parsers.py in __init__(self, src, **kwds)
       1872         if kwds.get("compression") is None and encoding:
       1873             if isinstance(src, str):
    -> 1874                 src = open(src, "rb")
       1875                 self.handles.append(src)
       1876 
    

    FileNotFoundError: [Errno 2] No such file or directory: ''


## Steps to consider

- [Pipelines](https://www.kaggle.com/alexisbcook/pipelines) - this makes missing values, categorical data transforming, choosing models, fitting the data and cross validation all simpler.
- [Missing Values](https://www.kaggle.com/alexisbcook/missing-values)
- [Categorical data](https://www.kaggle.com/alexisbcook/categorical-variables)
- [XGboost](https://www.kaggle.com/alexisbcook/xgboost)
- [Cross Validation](https://www.kaggle.com/alexisbcook/cross-validation)
- [Date Leakge](https://www.kaggle.com/alexisbcook/data-leakage)



```python
# kaggle reference (https://www.kaggle.com/alexisbcook/pipelines)  

# pipeline 
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

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
# This code relies on
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error # ideally cross validation, change later
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import cross_val_score
# then we'll need column transformers per dataset
# from sklearn.model_selection import train_test_split

y = df['']
features = ['', '', '', '', '' , '', '', '']
X = df[features]

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0)

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=6)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_val, y_val)], 
             verbose=False)

predictions = my_model.predict(X_val)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_val)))

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-3-2299590eb46f> in <module>
          7 # from sklearn.model_selection import train_test_split
          8 
    ----> 9 y = df['']
         10 features = ['', '', '', '', '' , '', '', '']
         11 X = df[features]
    

    NameError: name 'df' is not defined



```python
# I need to understand the looping check and if it can be used with xgboost and 
# how pipeline and cross validation works
# and perhaps better cat to num method
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


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-61b4b1234dea> in <module>
          4 
          5 le = preprocessing.LabelEncoder()
    ----> 6 le.fit(df.fruit)
          7 df['categorical_label'] = le.transform(df.fruit)
          8 
    

    NameError: name 'df' is not defined



```python

```
