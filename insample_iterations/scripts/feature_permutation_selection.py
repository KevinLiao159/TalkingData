import os
import psutil
import time
import numpy as np
import pandas as pd
import gc
# sklearn imports
from sklearn.metrics.scorer import roc_auc_score
from sklearn.utils import shuffle
import lightgbm
# klearn imports
import klearn.utils as gu

######################################
# Idea:
# 1. random permute each feature and get hold-out set score
# 2. compare hold-out scores, the permuted feature that causes the
#    biggest reduction in hold-out set score is the most important feature
# 3. the permuted feature that helps improve hold-out set score should be
#    removed from the feature set
######################################

# memory
t0 = time.time()
process = psutil.Process(os.getpid())
memused = process.memory_info().rss
print('Total memory in use before reading data: {:.02f} GB \
      '.format(memused/(2**30)))

##################################################
# read data
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
    'ip_channel_day_hour_clicks',
    'app_day_hour_clicks',
    # 'app_os_day_hour_clicks',
    # 'app_device_day_hour_clicks',
    # 'app_channel_day_hour_clicks',
    # 'ip_day_clicks',
    # 'ip_app_day_clicks',
    # 'ip_os_day_clicks',
    # 'ip_device_day_clicks',
    # 'ip_channel_day_clicks',
    # 'app_day_clicks',
    # 'app_os_day_clicks',
    # 'app_device_day_clicks',
    # 'app_channel_day_clicks',

    #  'ip_app_os_day_clicks',
    'ip_app_device_day_clicks',
    # 'ip_app_os_device_day_clicks',

    'ip_day_test_hh_clicks',
    # 'app_day_test_hh_clicks',

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
categorical_features = [
    'app',
    'device',
    'os',
    'channel',
    'hour',
    'in_test_hh',
]
print('done data prep!!!')
memused = process.memory_info().rss
print('Total memory in use after reading data: {:.02f} GB \
      '.format(memused/(2**30)))

##################################################
# get base score
##################################################
# prep data
dtrain = lightgbm.Dataset(
    df_train[features].values,
    label=df_train[target].values,
    feature_name=features,
    categorical_feature=categorical_features,
    free_raw_data=True,
)
# clean up
del df_train
gc.collect()
dtest = lightgbm.Dataset(
    df_test[features].values,
    label=df_test[target].values,
    feature_name=features,
    categorical_feature=categorical_features
)
# param
params = {
    'boosting_type': 'gbdt',
    # 'drop_rate': 0.09,
    'objective': 'binary',
    'learning_rate': 0.3,
    'num_leaves': 32,
    'max_depth': 5,
    'min_split_gain': 0,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.9,
    'min_child_samples': 100,
    'min_child_weight': 0,
    'max_bin': 100,
    'subsample_for_bin': 200000,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'scale_pos_weight': 99.7,
    'metric': 'auc',
    'nthread': 22,
    'verbose': 0,
}
model = lightgbm.train(
    params=params,
    train_set=dtrain,
    valid_sets=[dtrain, dtest],
    valid_names=['train', 'valid'],
    num_boost_round=1600,
    early_stopping_rounds=100,
    feature_name=features,
    categorical_feature=categorical_features,
    verbose_eval=1,
)
best_iter = model.best_iteration
proba = model.predict(df_test[features], num_iteration=best_iter)
roc_score = roc_auc_score(y_true=df_test[target], y_score=proba)
scores_dict = {'base': roc_score}

del model, dtrain, dtest, df_test
gc.collect()

##################################################
# permutation test
##################################################
for col in features:
    df_train = pd.read_hdf('../data/train.hdf').astype('float32')
    df_test = pd.read_hdf('../data/test.hdf').astype('float32')
    df_train[col] = shuffle(df_train[col], random_state=1).reset_index(drop=True)  # noqa
    df_test[col] = shuffle(df_test[col], random_state=1).reset_index(drop=True)
    # prep data
    dtrain = lightgbm.Dataset(
        df_train[features].values,
        label=df_train[target].values,
        feature_name=features,
        categorical_feature=categorical_features,
        free_raw_data=True,
    )
    # clean up
    del df_train
    gc.collect()
    model = lightgbm.train(
        params=params,
        train_set=dtrain,
        num_boost_round=best_iter,
        feature_name=features,
        categorical_feature=categorical_features,
        verbose_eval=1,
    )
    proba = model.predict(df_test[features], num_iteration=best_iter)
    roc_score = roc_auc_score(y_true=df_test[target], y_score=proba)
    scores_dict = {**scores_dict, **{col: roc_score}}
    del model, dtrain, df_test
    gc.collect()

##################################################
# save
##################################################
gu.save_object(scores_dict, 'permute_feature_score.pkl')
t1 = time.time()
t_min = np.round((t1-t0) / 60, 2)
print('It took {} mins to finish'.format(t_min))
