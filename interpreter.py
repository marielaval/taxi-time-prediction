# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm
import numpy as np

from preprocess import preprocess_final_data
from model import train_light_gbm

# Core functions


def cut_prediction_error(value):
    if value < -500:
        return '< -500'
    elif value < -300:
        return '-300'
    elif value < -200:
        return '< -200'
    elif value < -100:
        return '< -100'
    elif value < 0:
        return '< 0'
    elif value < 100:
        return '< 100'
    elif value < 200:
        return '< 200'
    elif value < 300:
        return '< 300'
    elif value < 500:
        return '< 500'
    else:
        return '> 500'


# Execution flow

df_train = preprocess_final_data(is_train=True)
df_test = preprocess_final_data(is_train=False)

final_lgb = train_light_gbm(df_train)


for col in df_train.columns.difference(df_test.columns):
    df_test[col] = np.zeros(df_test.shape[0])

df_test = df_test[df_train.columns]


lgb_test = df_test.drop("output_in_seconds",axis=1)


abs_error_lgbm = np.abs(df_test.output_in_seconds - final_lgb.predict(lgb_test))

deciles = np.percentile(abs_error_lgbm, [10, 90])


error = (df_test.output_in_seconds - final_lgb.predict(lgb_test)).apply(cut_prediction_error)


ax = sns.countplot(
    x=error,
    order=['< -500', '< -300', '< -200', '< -100', '< 0', '< 100', '< 200', '< 300', '< 500', '> 500']
)
ax.set(xlabel='Error = truth - prediction', ylabel='Count')
plt.savefig('output_distribution.png', dpi=300)



# Plot decision tree

lightgbm.plot_importance(final_lgb)


feature_imp = pd.DataFrame(
    sorted(zip(final_lgb.feature_importance(), df_train.drop("output_in_seconds",axis=1).columns)),
    columns=['Value', 'Feature']
)

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
