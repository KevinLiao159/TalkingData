import os
import psutil
import time
import gc
import numpy as np
import pandas as pd
import lightgbm
# sklearn imports
from sklearn.metrics.scorer import roc_auc_score

# memory status
process = psutil.Process(os.getpid())
memused = process.memory_info().rss
print('Total memory in use before reading data: {:.02f} GB '
      ''.format(memused / (2 ** 30)))

# # read data
df_train = pd.read_hdf('../data/train.hdf').astype('float32')
df_test = pd.read_hdf('../data/test.hdf').astype('float32')
# col
target = 'is_attributed'
features = [
    'app',
    'device',
    'os',
    'channel',
    'hour',
    'in_test_hh',
    'ip_day_hour_clicks',
    'ip_app_day_hour_clicks',
    'ip_os_day_hour_clicks',
    'ip_device_day_hour_clicks',
    'ip_day_test_hh_clicks',
    'ip_app_device_clicks',
    'ip_app_device_day_clicks',
    'ip_day_nunique_app',
    'ip_day_nunique_device',
    'ip_day_nunique_channel',
    'ip_day_nunique_hour',
    'ip_nunique_app',
    'ip_nunique_device',
    'ip_nunique_channel',
    'ip_nunique_hour',
    'app_day_nunique_channel',
    'app_nunique_channel',
    'ip_app_day_nunique_os',
    'ip_app_nunique_os',
    'ip_device_os_day_nunique_app',
    'ip_device_os_nunique_app',
    'ip_app_day_var_hour',
    'ip_device_day_var_hour',
    'ip_os_day_var_hour',
    'ip_channel_day_var_hour',
    'ip_app_os_var_hour',
    'ip_app_channel_var_day',
    'ip_app_channel_mean_hour',
    'ip_day_cumcount',
    'ip_cumcount',
    'ip_app_day_cumcount',
    'ip_app_cumcount',
    'ip_device_os_day_cumcount',
    'ip_device_os_cumcount',
    'next_click',
    'previous_click',
]
# categorical
categorical_features = [
    'app',
    'device',
    'os',
    'channel',
    'hour',
    'in_test_hh',
]
# prep data
dtrain = lightgbm.Dataset(
    df_train[features].values,
    label=df_train[target].values,
    feature_name=features,
    categorical_feature=categorical_features,
    free_raw_data=True,
)
dtest = lightgbm.Dataset(
    df_test[features].values,
    label=df_test[target].values,
    feature_name=features,
    categorical_feature=categorical_features
)
# cleanup
del df_train
gc.collect()
print('done data prep!!!')
# memory status
memused = process.memory_info().rss
print('Total memory in use after reading data: {:.02f} GB '
      ''.format(memused / (2 ** 30)))

###################################################################
params = {
    'boosting_type': 'gbdt',            # I think dart would be better, but takes too long to run # noqa
    # 'drop_rate': 0.09,                # only used in dart, Rate at which to drop trees # noqa
    'objective': 'binary',
    'learning_rate': 0.1,
    'num_leaves': 32,                   # Was 255: Reduced to control overfitting # noqa
    'max_depth': 6,                     # Was 8: LightGBM splits leaf-wise, so control depth via num_leaves # noqa
    'min_split_gain': 0,
    'subsample': 0.9,                   # Was 0.7
    'subsample_freq': 1,
    'colsample_bytree': 0.7,
    'min_child_samples': 100,
    'min_child_weight': 0,
    'max_bin': 100,                     # default was 255
    'subsample_for_bin': 200000,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'scale_pos_weight': 100,
    'metric': 'auc',
    'nthread': 22,
    'verbose': 0,
# seed is set default
}
# train
t0 = time.time()
model = lightgbm.train(
    params=params,
    train_set=dtrain,
    valid_sets=[dtrain, dtest],
    valid_names=['train', 'valid'],
    num_boost_round=350,
    early_stopping_rounds=30,
    feature_name=features,
    categorical_feature=categorical_features,
    verbose_eval=1,
    # init_model='model.txt'
)
####################################################################
t1 = time.time()
t_min = np.round((t1-t0) / 60, 2)
print('It took {} mins to train model'.format(t_min))
####################################################################
proba = model.predict(df_test[features], num_iteration=model.best_iteration)
roc_score = roc_auc_score(y_true=df_test[target], y_score=proba)
print('Out of sample roc score is {}'.format(roc_score))
