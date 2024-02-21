# 2024-02-20
# Peter R.
# XGBoost script
# Positive breaks, n_estimators (number of trees)=1000 and with optimal parameter from DRAC model_bp1 & early stopping

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
print("positive breaks")

# Windows
#df1 = pd.read_csv(r'.\data\forest_evi_breaks_positive_sam1.csv', skipinitialspace=True)
# DRAC
df1 = pd.read_csv(r'./data/forest_evi_breaks_positive_v2.csv', skipinitialspace=True)
#df1.head()

df2 = pd.get_dummies(df1, columns=['for_pro'], dtype=float)

#df2= df2[df2['precipitation'].notna()]

#X1 = df2.iloc[:,2:30]

#X1.drop(X1.columns[[0:5, 7:10, 12, 15, 16, 18, 19, 20,21,22,23,24,25]], axis=1,inplace=True)

#cols2 = ['for_age','for_con', 'cmi_sm', 'cmi_sm_lag1', 'cmi_sm_lag2', 'cmi_sm_lag3', 'dd5_wt', 'nffd_wt', 'nffd_wt_lag1', 'nffd_wt_lag2', 'nffd_wt_lag3', 'pas_sm', 'pas_sm_lag1', 'pas_sm_lag2', 'pas_sm_lag3', 'for_pro_0']
#cols2 = ['for_age','for_con', 'cmi_sm', 'cmi_sm_lag1', 'cmi_sm_lag2', 'cmi_sm_lag3', 'dd5_wt', 'nffd_wt', 'nffd_wt_lag1', 'nffd_wt_lag2', 'nffd_wt_lag3', 'pas_sm', 'pas_sm_lag1', 'pas_sm_lag2', 'pas_sm_lag3', 'for_pro_0', 'map', 'map_lag1', 'map_lag2', 'map_lag3','mat', 'mat_lag1', 'mat_lag2', 'mat_lag3','rh', 'rh_lag1', 'rh_lag2', 'rh_lag3']
cols2 =  ['for_con', 'cmi_sm', 'cmi_sm_lag1', 'cmi_sm_lag2', 'cmi_sm_lag3', 'dd5_wt', 'dd5_wt_lag2']
#cols2 = ['for_age','for_con', 'cmi_sm', 'cmi_sm_lag1', 'cmi_sm_lag2', 'cmi_sm_lag3', 'dd5_wt', 'dd5_wt_lag2', 'for_pro_0', 'for_pro_1']
X1 = df2[cols2]

# vars used in previous version: 'map', 'map_lag1', 'map_lag2', 'map_lag3','mat', 'mat_lag1', 'mat_lag2', 'mat_lag3','rh', 'rh_lag1', 'rh_lag2', 'rh_lag3'
#features_names1 = ["age","deciduous","elevation","precipitation","temperature","precipitation_lag1", "temperature_lag1", "precipitation_lag2", "temperature_lag2", "precipitation_lag3", "temperature_lag3",
#                 "rh" ,"rh_lag1","rh_lag2","rh_lag3"]

y1 = df2.iloc[:,6]

# Fine tune parameters using RandomizedSearchCV (faster)
# max_depth is tree complexity in Elith et al. 2008
# n_estimators=100 is the number of trees. Elith et al. 2008 say this should be 1000 at least
# Elith et al. 2008 suggests low learning rate

seed = 7 # random seed to help with replication
testsize1 = 0.33 # percent of records to test after training

# Split data set. Note the 'stratify' option
x1_train, x1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=testsize1, random_state=seed)



#{'reg_lambda': 10, 'reg_alpha': 0.1, 'objective': 'reg:squarederror', 'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.005, 'gamma': 0.05}

#{'reg_lambda': 10, 'reg_alpha': 1, 'objective': 'reg:squarederror', 'n_estimators': 1000, 'max_depth': 8, 'learning_rate': 0.01, 'gamma': 0.2}
            
model_bp2 = XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, early_stopping_rounds=50,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=0.2, gpu_id=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=0.01, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=8, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             n_estimators=1000, n_jobs=None, num_parallel_tree=None,
             predictor=None, random_state=42, reg_lambda=10, reg_alpha=1)



# EVALUATION (with test)
eval_set = [(x1_train, y1_train), (x1_test, y1_test)]
#UserWarning: `eval_metric` in `fit` method is deprecated for better compatibility with scikit-learn, use `eval_metric` in constructor or`set_params` instead.
model_bp2.fit(x1_train, y1_train, eval_set=eval_set, verbose=False)
# make predictions for test data
y_pred = model_bp2.predict(x1_test)
predictions = [round(value) for value in y_pred]
# retrieve performance metrics
results = model_bp2.evals_result()

mse = mean_squared_error(y1_test, y_pred)
#r2 = explained_variance_score(y1_test, ypred)
r2 = r2_score(y1_test, y_pred)
print("MSE: %.2f" % mse)

print("RMSE: %.2f" % (mse**(1/2.0)))

print("R-sq: %.2f" % r2)

# Save model
# save in JSON format
#model_bp1.save_model("model_bp1_pos_brks_v1.json")
#model_bp2.save_model("model_bp2_pos_brks_v3.json") # Take 2 with new climate vars
#model_bp2.save_model("model_bp2_pos_brks_v4.json") # Take 2 with new climate vars + old vars
model_bp2.save_model("model_bp2_pos_brks_v5.json") # Take 3 with VIF var subset
#model_bp2.save_model("model_bp2_pos_brks_v6.json") # Take 3 with VIF var subset + a few old vars (for_age, etc.)
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