# Libraries

import lightgbm as lgb

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocess import preprocess_final_data
import time
import numpy as np


#######################    LGBM   #######################

# Constant
random_state = 42
test_size = 0.1
early_stopping_rounds = 50
params = {'num_leaves': 10,
         'learning_rate': 0.01,
         'subsample': 0.9,
         'colsample_bytree': 0.7,
         'lambda_l1': 1,
         'boosting_type': 'gbdt',
         'max_depth': 4,
         'subsample_for_bin': 50000,
         'objective': 'regression',
         'subsample_freq': 1,
         'seed': 17,
         'metric': 'rmse',
         'max_bin': 100,
         'n_estimators': 1000,
         'min_data_in_leaf':1000}


# Core functions

def prepare_lgbm_datasets(df_train, target_column):
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop(target_column, axis=1),
                                                        df_train[target_column],
                                                        test_size=test_size,
                                                        random_state=random_state)
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_eval = lgb.Dataset(X_test, label=y_test)

    return lgb_train, lgb_eval


def train_light_gbm(df_train):
    lgb_train, lgb_eval = prepare_lgbm_datasets(df_train, 'output_in_seconds')

    final_lgb = lgb.train(params, lgb_train, early_stopping_rounds=early_stopping_rounds, valid_sets=lgb_eval)

    return final_lgb


# Execution flow

df_train = preprocess_final_data(is_train=True)
df_test = preprocess_final_data(is_train=False)

final_lgb = train_light_gbm(df_train)

for col in df_train.columns.difference(df_test.columns):
    df_test[col] = np.zeros(df_test.shape[0])

df_test = df_test[df_train.columns]

lgb_test = df_test.drop("output_in_seconds",axis=1)
y_test = df_test.output_in_seconds
y_pred_lgb = final_lgb.predict(lgb_test)

rmse_lgb = np.sqrt(mean_squared_error(y_test,y_pred_lgb))

print("Test RMSE LGBM: %s seconds ---" % rmse_lgb)


################# Linear REGRESSION  ###########################

df_test = preprocess_final_data(is_train=False)

df_train = df_train.drop(df_train.columns.difference(df_test.columns), axis=1)

# constants

target_column = 'output_in_seconds'
random_state = 42
test_size = 0.33

# create evaluation set

X_train, X_eval, y_train, y_eval = train_test_split(df_train.drop(target_column, axis=1),
                                                        df_train[target_column],
                                                        test_size=test_size,
                                                        random_state=random_state)

# train the model

start_time = time.time()
reg = LinearRegression().fit(X_train, y_train)
print("training time: %s seconds ---" % (time.time() - start_time))

# predict on evaluation set

start_time = time.time()
prediction_eval = reg.predict(X_eval)
print("prediction time on eval set: %s seconds ---" % (time.time() - start_time))

RMSE_eval = np.sqrt(mean_squared_error(y_eval, prediction_eval))

print("Eval RMSE: %s seconds ---" % RMSE_eval)

# predict on test set

start_time = time.time()
prediction_test = reg.predict(df_test.drop(target_column, axis=1))
print("prediction time on test set: %s seconds ---" % (time.time() - start_time))

RMSE_test = np.sqrt(mean_squared_error(df_test[target_column], prediction_test))

print("Test RMSE: %s seconds ---" % RMSE_test)
