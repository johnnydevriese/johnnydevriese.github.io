---
categories: kaggle machine-learning
date: '2023-02-09'
layout: post
slug: theneedledrop-ratings
title: Theneedledrop Ratings
---

# Decision Trees (CART)

Decision trees are defined by recursively partitioning the input space, and defining a local model in
each resulting region of input space. The overall model can be represented by a tree, with one leaf
per region.

(image removed because Kaggle was complaininga about size limits...)

An ensemble method is using building a tree with multiple decision tress.

(image removed because Kaggle was complaininga about size limits...)

# Boosting

Ensemble of tress has a model of the form

$$ f(x; \theta) = \sum_{m =1}^{M} \beta_m F_m(x ; \theta_m) $$

where $F_m$ is the $m$th tree and $\beta_m$ is the weight. We know that if $F_m$ is even *slightly* better than 50% accurate then the final $f$ becomes very accurate. So, an ensemle of **weak learners** becomes a **strong learner**.

# Gradient Boosting

If we assume our **weak learner** is a regression tree then then our model has the form

$$ F_m(x) = \sum_{j =1}^{J_M} \omega_{jm} \prod(x \in R_{jm}) $$

where $w_{jm}$ is the predicted output for reino $R_{jm}$. 


# XGBoost - eXtreme Gradient Boosting

Is a very popular implementation of gradient boosted trees and can be used for both classification and regression problems. It adds some tricks though 

* Adds a regularizer on complexity of the tree. 
* Uses second order approximation forr the loss (Taylor Series expansion) instead of linear approximation. 
* It samples features at internal nodes. 
* Adds computer science tricks for scalability to very large datasets.

The regularized objective function to optimize is 

$$Obj(\Theta) = L(\Theta) + \Omega(\Theta)$$

Where $L(\Theta)$ is the **Training Loss** which measures how well the model fits the training data and $\Omega(\Theta)$ is the **Regularization** which measures the complexity of the model. This can also be written as 

$$ L(f) = \sum_{i = 1}^N l (y_i, f(x_i)) + \Omega(f)$$

Where the regularization term can be written as

$$ \Omega(f) = \gamma J + \frac{1}{2} \lambda \sum_{j = 1}^{J} w_j^2$$

Whjere $J$ is the *number of leaves* and $\gamma \geq 0$ and $ \lambda \geq 0$ are regularization coefficients at the $m$'th step. 


### The two components of the objective function are important 

* If we optimize for **training loss** then we end up with a stronger *predictive* model. 

* If we optimize for **regularization** we end up with *simpler* models 


References: Tianqi Chen "Introduction to Boosted Trees" 2014 and Kevin Murphy "Probabilistic Machine Learning" 2022


```python
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib
#matplotlib.use('nbagg')
import matplotlib.pyplot as plt
%matplotlib inline

from IPython.display import Markdown, display, HTML
def printmd(string):
    display(Markdown(string))

# SKLEARN - ML
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import MinMaxScaler

# TENSORFLOW - ML
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.callbacks import EarlyStopping
```

# What is @theneedledrop ? 

Anthony Fantano runs a (relatively) popular YouTube channel where he reviews albums. Many people seem to believe that he's relatively objective and that his reviews are generally Gaussian and center around 5. So I went to reddit and found a dataset with all of his reviews and then later found the 160k track Spotify (now seemingly defunct) and thought about joining the two dataset to see if there were any underlying patterns to the Melon's madness. 


```python
ratings_df = pd.read_csv("../input/needle-drop-ratings-2021/needle_drop_all_ratings.csv", header=2)
```


```python
print(ratings_df.describe())
print(ratings_df.count())
```

# We have a lot of cleaning up to do! 

Convert `SCORES` of `NOT GOOD` to 0 and `CLASSIC` to 10 and many of the others are a bit at my discretion. 


```python
ratings_df.loc[ratings_df["SCORE"] == "NOT GOOD", "SCORE"] = '0'
ratings_df.loc[ratings_df["SCORE"] == "CLASSIC", "SCORE"] = '10'
ratings_df.loc[ratings_df["SCORE"] == "meh", "SCORE"] = '0'
ratings_df.loc[ratings_df["SCORE"] == "based", "SCORE"] = '10'
ratings_df.loc[ratings_df["SCORE"] == "awesomely nasty", "SCORE"] = '10'
ratings_df.loc[ratings_df["SCORE"] == "notbad", "SCORE"] = '5' 
ratings_df.loc[ratings_df["SCORE"] == "lukewarm", "SCORE"] = '5'
ratings_df.loc[ratings_df["SCORE"] == "disappointing", "SCORE"] = "2"
ratings_df.loc[ratings_df["SCORE"] == "solid", "SCORE"] = "8"
ratings_df.loc[ratings_df["SCORE"] == "it's solid", "SCORE"] = "8"
ratings_df.loc[ratings_df["SCORE"] == 'classic', "SCORE"] = "10"
ratings_df.loc[ratings_df["SCORE"] == 'dime-a-dozen', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == 'really inconsistent comp, pick out the best tracks', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == "banging", "SCORE"] = "9"
ratings_df.loc[ratings_df["SCORE"] == 'PGOODMAYBEA7', "SCORE"] = "7"
ratings_df.loc[ratings_df["SCORE"] == 'b.o.b. album', "SCORE"] = "5" # not sure what b.o.b. means lol 
ratings_df.loc[ratings_df["SCORE"] == 'opinion', "SCORE"] = "5" # not sure what to cast this to 
ratings_df.loc[ratings_df["SCORE"] == 'MYSCOREWHENTHEPU$$YISTOOGOOD', "SCORE"] = "7"
ratings_df.loc[ratings_df["SCORE"] == 'whitegirlturnupmusic', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == 'BUTTSECKS', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == '6(66)', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == 'http://www.runthejewels.net', "SCORE"] = "10"
ratings_df.loc[ratings_df["SCORE"] == 'no', "SCORE"] = "0"
ratings_df.loc[ratings_df["SCORE"] == 'burp', "SCORE"] = "0"
ratings_df.loc[ratings_df["SCORE"] == '???', "SCORE"] = "0"
ratings_df.loc[ratings_df["SCORE"] == 'uhhhhhhhhh', "SCORE"] = "0"
ratings_df.loc[ratings_df["SCORE"] == 'bread', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == 'f', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == '[Chris] 7', "SCORE"] = "7"
ratings_df.loc[ratings_df["SCORE"] == '[he just ate]', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == '[he threw up]', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == '[Deerhoof won]', "SCORE"] = "5"
ratings_df.loc[ratings_df["SCORE"] == '[he discussed copyright]', "SCORE"] = "5"
```


```python
print(type(ratings_df))
```

Turns out we have around 40 albums/mixtapes without a score. 


```python
# TODO: refactor this logic 

ratings_df_filtered = ratings_df[ratings_df['SCORE'] != 'none']
ratings_df_filtered = ratings_df_filtered[ratings_df['SCORE'] != 'NONE']
ratings_df_filtered = ratings_df_filtered[ratings_df['SCORE'] != 'UNKNOWN']


# df_none.head()
# print(df_none.to_string())
```


```python
# show the data types
# of each columns
print (ratings_df.dtypes)

# convert string to an integer
ratings_df_filtered['SCORE'] = ratings_df_filtered['SCORE'].astype(int)

# print (ratings_df['SCORE'].dtypes)
print (ratings_df_filtered.dtypes)
```


```python
sns.histplot(data=ratings_df_filtered, x=ratings_df_filtered["SCORE"])
```


```python
# only care about numerics 
ratings_df = ratings_df[ratings_df['SCORE'].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])]
print(type(ratings_df))
ratings_series = ratings_df['SCORE'].astype('int32')
print(type(ratings_series))
ratings_series.hist()
# sns.histplot(data=ratings_series)
```

We can see this looks like a Gaussian that is shifted to around 7. Even though manty people online like to comment that Fantano tries to apply a normal distribution centered around 5 to his ratings. This clearly is not the case. 

TODO: add different Gaussians plots with params


```python
ratings_df.loc[ratings_df["SCORE"] == '0'].count()
```


```python
# show the data types
# of each columns
print (ratings_df_filtered.dtypes)

# convert string to an integer
ratings_df_filtered['SCORE'] = ratings_df_filtered['SCORE'].astype(int)

# print (ratings_df['SCORE'].dtypes)
print (ratings_df_filtered.dtypes)

```

# Import Spotify dataset

### Spotify Dataset Feature Summary:
Dataset with ~1.3M observations, 19 columns (13 numerical and 6 categorical/dummy) and no nan values
numerical columns:

* acousticness: The relative metric of the track being acoustic, (Ranges from 0 to 1)
* danceability: The relative measurement of the track being danceable, (Ranges from 0 to 1)
* energy: The energy of the track, (Ranges from 0 to 1)
* duration_ms: The length of the track in milliseconds (ms), (Integer typically ranging from 200k to 300k)
* instrumentalness:, The relative ratio of the track being instrumental, (Ranges from 0 to 1)
* valence: The positiveness of the track, (Ranges from 0 to 1)
* popularity: The popularity of the song lately, default country = US, (Ranges from 0 to 100)
* tempo:The tempo of the track in Beat Per Minute (BPM), (Float typically ranging from 50 to 150)
* liveness: The relative duration of the track sounding as a live performance, (Ranges from 0 to 1)
* loudness: Relative loudness of the track in decibel (dB), (Float typically ranging from -60 to 0)
* speechiness: The relative length of the track containing any kind of human voice, (Ranges from 0 to 1)
* year: The release year of track, (Ranges from 1921 to 2020)
id, The primary identifier for the track, generated by Spotify


### categorical columns:

* key: The primary key of the track encoded as integers in between 0 and 11 (starting on C as 0, C# as 1 and so on…)
* artists: The list of artists credited for production of the track
* release_date: Date of release mostly in yyyy-mm-dd format, however precision of date may vary
* name: The title of the track
* mode: The binary value representing whether the track starts with a major (1) chord progression or a minor (0)
explicit: The binary value whether the track contains explicit content or not, (0 = No explicit content, 1 = Explicit content)


```python
tracks_df = pd.read_csv("../input/spotify-12m-songs/tracks_features.csv")
```


```python
tracks_df.head()
```

# TODO: We do not want to group by album mean -- this was a bad idea! 


```python
# album_df = tracks_df.groupby('album').mean()
# album_df = album_df.reset_index()
```


```python
# merge the two datasets on their album name 
# rating_albums_df = ratings_df.merge(album_df, left_on='ALBUM TITLE', right_on='album')
```


```python
# rating_albums_df["SCORE"].count()
```

# Updated: We want to just merge/join our ratings with the track if we have it
Note that the Spotify dataset doesnt' have all of the albums that Fantano has. We should analyze which albums and how many albums we do have. 


```python
ratings_df_filtered.count()
```


```python
rating_albums_df = ratings_df_filtered.merge(tracks_df, left_on='ALBUM TITLE', right_on='album')
```


```python
rating_albums_df.dtypes
```


```python
rating_albums_df.count()
```


```python
rating_albums_df.groupby('album').count()
# 537 rows 
```


```python
plt.figure(figsize = (15,15)) #creating the 'canvas'
sns.heatmap(tracks_df.corr(), annot=True)
```


```python
rating_albums_df['score'] = rating_albums_df['SCORE'].astype(str).astype(int)
rating_albums_df.drop(['ARTISTS', 'ALBUM TITLE', 'SCORE', 'Unnamed: 3', 'album'], axis=1)
```


```python
feature_names = [
       'track_number', 'disc_number', 'explicit', 'danceability', 'energy',
       'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
       'time_signature', 'year']

X = pd.DataFrame(rating_albums_df, columns=feature_names)
y = rating_albums_df['score']
```


```python
y
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

# XGBoost's hyperparameters

At this point, before building the model, you should be aware of the tuning parameters that XGBoost provides. Well, there are a plethora of tuning parameters for tree-based learners in XGBoost and you can read all about them here. But the most common ones that you should know are:

* **learning_rate**: step size shrinkage used to prevent overfitting. Range is [0,1]
* **max_depth**: determines how deeply each tree is allowed to grow during any boosting round.
* **subsample**: percentage of samples used per tree. Low value can lead to underfitting.
* **colsample_bytree**: percentage of features used per tree. High value can lead to overfitting.
* **n_estimators**: number of trees you want to build.
* **objective**: determines the loss function to be used like reg:linear for regression problems, reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability.


### XGBoost also supports regularization parameters to penalize models as they become more complex and reduce them to simple (parsimonious) models.

* **gamma**: controls whether a given node will split based on the expected reduction in loss after the split. A higher value leads to fewer splits. Supported only for tree-based learners.
* **alpha**: L1 regularization on leaf weights. A large value leads to more regularization.
* **lambda**: L2 regularization on leaf weights and is smoother than L1 regularization.



```python
model = xgb.XGBClassifier(n_estimators=10000,)
```


```python
model.fit(X_train, y_train)
```


```python
y_pred_basic = model.predict(X_test)
```


```python
type(y_pred_basic)
```


```python
y_pred = y_pred_basic.astype(int)

y_test_1 = y_test.astype(int)
```


```python
from sklearn import metrics

metrics.accuracy_score(y_test_1, y_pred)
# macro_averaged_f1 = metrics.f1_score(y_test_1, y_pred, average = 'macro')

# print(macro_averaged_f1)
```

So we're correct about a quarter of the time. Which isn't that great but it isn't terrible either for such a small dataset. But with Bayesian Optimization we increased our accuracy by 8 percent! 


```python
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score


pbounds = {
    'learning_rate': (0.01, 1.0),
    'n_estimators': (100, 1000),
    'max_depth': (3,100),
    'subsample': (1.0, 1.0),  # Change for big datasets
    'colsample': (1.0, 1.0),  # Change for datasets with lots of features
    'gamma': (0, 5)}

def xgboost_hyper_param(learning_rate,
                        n_estimators,
                        max_depth,
                        subsample,
                        colsample,
                        gamma):

    max_depth = int(max_depth)
    n_estimators = int(n_estimators)

    clf = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma)
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
#     return np.mean(cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc'))
    return metrics.accuracy_score(y_test, y_pred)
#     return metrics.accuracy_score(X_train, y_train)

optimizer = BayesianOptimization(
    f=xgboost_hyper_param,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(n_iter=15, init_points=2)
```


```python
#TODO: maybe compare regression vs classification with bayesian optimization? 
```
