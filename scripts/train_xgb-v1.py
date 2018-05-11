import os
import psutil
import time
import gc
import numpy as np
import pandas as pd
import xgboost

# memory status
process = psutil.Process(os.getpid())
memused = process.memory_info().rss
print('Total memory in use before reading data: {:.02f} GB '
      ''.format(memused / (2 ** 30)))

# read data
df_train = pd.read_hdf('../data/train_v1.hdf')
# col
target = 'is_attributed'
features = [
    'app',
    'device',
    'os',
    'channel',
    'dow',
    'doy',
    'ip_clicks',
    'ip_app_comb_clicks',
    'hour',
    'app_clicks',
    'os_clicks',
    'device_clicks',
    'app_device_comb_clicks',
    'app_os_comb_clicks',
]

# prep data
dtrain = xgboost.DMatrix(df_train[features], df_train[target])
del df_train
gc.collect()
print('done data prep!!!')
# memory status
memused = process.memory_info().rss
print('Total memory in use after reading data: {:.02f} GB '
      ''.format(memused / (2 ** 30)))

t0 = time.time()
###################################################################
params = {
    'objective': 'binary:logistic',
    'tree_method': "hist",
    'grow_policy': "lossguide",
    'max_leaves': 1400,
    'eta': 0.1,
    'max_depth': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.9,
    'min_child_weight': 0,
    'alpha': 4,
    'lambda': 1,
    'scale_pos_weight': 9,
    'eval_metric': 'auc',
    'nthread': 18,
    'random_state': 99,
    'silent': True
}
# train
model = xgboost.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=145,
    maximize=True,
    verbose_eval=1
)
####################################################################
t1 = time.time()
t_min = np.round((t1-t0) / 60, 2)
print('It took {} mins to train model'.format(t_min))

# clean up
del dtrain
gc.collect()
####################################################################
# submit
df_test = pd.read_hdf('../data/test_v1.hdf')
# pred
df_test['is_attributed'] = model.predict(xgboost.DMatrix(df_test[features]))
# create submission
sub_cols = ['click_id', 'is_attributed']
df_sub = df_test[sub_cols]
# save
df_sub.to_csv('../data/submission_v1.csv', index=False)
