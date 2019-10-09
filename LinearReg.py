from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from preprocess import preprocess_final_data

import time
import pandas as pd
import numpy as np

# create train and test dataframes with the same columns

df_train = preprocess_final_data(is_train=True)

df_test = preprocess_final_data(is_train=False)

for col in df_train.columns.difference(df_test.columns):
    df_test[col] = np.zeros(df_test.shape[0])

df_test = df_test[df_train.columns]

# df_train = df_train.drop(df_train.columns.difference(df_test.columns), axis=1)

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

abs_error = np.abs(df_test.output_in_seconds - reg.predict(df_test.drop(target_column, axis=1)))

deciles = np.percentile(abs_error, [10, 90])
