# 2023-08-25
# Peter R.
# XGBoost script
# Positive breaks, n_estimators (number of trees)=1000, all breaks

import os
import time

import pandas as pd
from numpy import nan
import xgboost as xgb
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


start = time.time()

# Get the current working directory
cwd = os.getcwd()

#print(cwd)

# DRAC directory
os.chdir("/home/georod/projects/def-mfortin/georod/scripts/github/forc_trends/models/xgboost")

print("XGB version:", xgb.__version__)
print("all breaks")

# Windows
#df1 = pd.read_csv(r'.\data\forest_evi_breaks_positive_sam1.csv', skipinitialspace=True)
# DRAC
df1 = pd.read_csv(r'./data/forest_evi_breaks_v1.csv', skipinitialspace=True)
#df1.head()

df2 = pd.get_dummies(df1, columns=['protected'], dtype=float)

df2= df2[df2['precipitation'].notna()]

X1 = df2.iloc[:,2:24]

X1.drop(X1.columns[[2, 12, 14, 16, 18]], axis=1,inplace=True)

y1 = df2.iloc[:,1]


features_names1 = ["age","deciduous","elevation","precipitation","temperature","precipitation_lag1", "temperature_lag1", "precipitation_lag2", "temperature_lag2", "precipitation_lag3", "temperature_lag3",
                 "rh" ,"rh_lag1","rh_lag2","rh_lag3"]


# Fine tune parameters using RandomizedSearchCV (faster)
# max_depth is tree complexity in Elith et al. 2008
# n_estimators=100 is the number of trees. Elith et al. 2008 say this should be 1000 at least
# Elith et al. 2008 suggests low learning rate

seed = 7 # random seed to help with replication
testsize1 = 0.33 # percent of records to test after training

# Split data set. Note the 'stratify' option
x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=testsize1, random_state=seed)


# Fine tunning parameters with Random Search
#search space
params_xgboost = {
 "learning_rate"    : [ 0.001, 0.005, 0.01, 0.05, 0.10, 0.15],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10],
 "gamma"            : [ 0.0, 0.01, 0.05, 0.1, 0.2, 0.3],
 #"colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7],
 #'n_estimators'     : [5, 10, 15, 20, 25, 30, 35],
'n_estimators'     : [1000],
 'objective': ['reg:squarederror'],
#'early_stopping_rounds': [10]
# reg_alpha provides L1 regularization to the weight, higher values result in more conservative models
"reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
# reg_lambda provides L2 regularization to the weight, higher values result in more conservative models
"reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100]
}

model_base1 = XGBRegressor()

random_search = RandomizedSearchCV(estimator = model_base1, 
                      param_distributions = params_xgboost, 
                      n_iter = 100, 
                      cv = 5, 
                      verbose=10, 
                      random_state=42,
                      scoring = 'neg_mean_squared_error', 
                        refit=True,
                      n_jobs = -1)

#params glare proba
random_search.fit(x1_train, y1_train)

#random_search
print(random_search.best_params_)
print(random_search.best_estimator_)

# How to early_stopping_rounds=10?
# Model with best parameters 1
model_bp1 = random_search.best_estimator_

#print(model_bp1)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=seed)

# evaluate model with train
# -1 means using all processors in parallel
# cross val takes place withing the train data set
scores = cross_val_score(model_bp1, x1_train, y1_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

scores2 = cross_val_score(model_bp1, x1_train, y1_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

# force scores to be positive
scores2 = absolute(scores)
print('Mean MSE: %.3f (%.3f)' % (scores2.mean(), scores2.std()) )

print('Mean RMSE: %.3f (%.3f)' % (scores2.mean()**(1/2.0), scores2.std()**(1/2.0)))

# evaluate model with variance explained
scores3 = cross_val_score(model_bp1, x1_train, y1_train, scoring='explained_variance', cv=cv, n_jobs=-1)
#print(scores3)

# force scores to be positive
#print(statistics.mean(scores3))
print('Mean Var. Explained: %.3f (%.3f)' % (scores3.mean(), scores3.std()) ) 

# R-squared
# evaluate model with variance explained
scores4 = cross_val_score(model_bp1, x1_train, y1_train, scoring='r2', cv=cv, n_jobs=-1)
#print(scores3)

# force scores to be positive
#print(statistics.mean(scores3))
print('R-sq: %.3f (%.3f)' % (scores4.mean(), scores4.std()) ) 


# EVALUATION (with test)
eval_set = [(x1_train, y1_train), (x1_test, y1_test)]
#UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
model_bp1.fit(x1_train, y1_train, eval_set=eval_set, verbose=False)
# make predictions for test data
y_pred = model_bp1.predict(x1_test)
predictions = [round(value) for value in y_pred]
# retrieve performance metrics
results = model_bp1.evals_result()

mse = mean_squared_error(y1_test, y_pred)
#r2 = explained_variance_score(y1_test, ypred)
r2 = r2_score(y1_test, y_pred)
print("MSE: %.2f" % mse)

print("RMSE: %.2f" % (mse**(1/2.0)))

print("R-sq: %.2f" % r2)

# Save model
# save in JSON format
model_bp1.save_model("model_bp1_all_brks_v1.json")
# save in text format
#model_m2.save_model("model_m2.txt")

end = time.time()

total_time = end-start
#total_time
print("Total time: %.2f" % total_time)

# Load model
# load saved model
#model2 = xgb.Regressor()
#model2.load_model("model_regression1.json")