import time
import gc
import numpy as np
import pandas as pd
import operator
# sklearn imports
from sklearn.metrics.scorer import roc_auc_score
import xgboost
# klearn imports
import klearn.utils as gu

######################################
# Idea:
# 1. include only base features
# 2  adding a feature with the most improvement in out-of-sample score once a time  # noqa
# 3. we can see that out-of-sample score starts to decrease after peaking at a set of features # noqa
# 4. prune those features after that peak
######################################
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
    'dow',
    'doy',
    'ip_clicks'
]
# get new features
new_features = [
    'ip_app_comb_clicks', 'ip_device_comb_clicks', 'app_device_comb_clicks',
    'ip_channel_comb_clicks', 'app_os_comb_clicks', 'app_clicks',
    'ip_os_comb_clicks', 'hour', 'app_hour_comb_clicks',
    'device_hour_comb_clicks', 'ip_hour_comb_clicks', 'device_clicks',
    'app_channel_comb_clicks', 'os_device_comb_clicks', 'os_clicks',
]

# set model parap
params = {
    'objective': 'binary:logistic',
    'tree_method': "hist",
    'grow_policy': "lossguide",
    'max_leaves': 1400,
    'eta': 0.3,
    'max_depth': 0,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'min_child_weight': 0,
    'alpha': 4,
    'lambda': 1,
    'scale_pos_weight': 9,
    'eval_metric': 'auc',
    'nthread': 18,
    'random_state': 99,
    'silent': True
}
# get base line score
# prep data
dtrain = xgboost.DMatrix(df_train[features + new_features], df_train[target])
# train
model = xgboost.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=30,
    maximize=True,
    verbose_eval=1
)
# predict proba
proba = model.predict(xgboost.DMatrix(df_test[features + new_features]))
roc_score = roc_auc_score(y_true=df_test[target], y_score=proba)
# clean up
del dtrain, model, proba
gc.collect()
# score baseline
score_list = [(0, {"base": roc_score})]
i = 0
while len(new_features):
    i += 1
    score_dict_temp = {}
    for feature_add in new_features:
        # prep data
        dtrain = xgboost.DMatrix(df_train[features + [feature_add]], df_train[target])    # noqa
        print('done data prep!!!')
        t0 = time.time()
        ###################################################################
        # train
        model = xgboost.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=30,
            maximize=True,
            verbose_eval=1
        )
        ####################################################################
        t1 = time.time()
        t_min = np.round((t1-t0) / 60, 2)
        print('It took {} mins to train model'.format(t_min))
        ####################################################################
        # predict proba
        proba = model.predict(xgboost.DMatrix(df_test[features + [feature_add]]))          # noqa
        roc_score = roc_auc_score(y_true=df_test[target], y_score=proba)
        print('Out of sample roc score is {} removing {}'.format(roc_score, feature_add))  # noqa
        score_dict_temp = {**score_dict_temp, **{feature_add: roc_score}}
        # clean up
        del dtrain, model, proba
        gc.collect()
    #####################################################################
    # get the max score
    feature_to_add = max(score_dict_temp.items(), key=operator.itemgetter(1))[0]           # noqa
    feature_add_score = max(score_dict_temp.items(), key=operator.itemgetter(1))[1]        # noqa
    # update
    score_list.append((i, {feature_to_add: feature_add_score}))
    new_features.remove(feature_to_add)

# save score
gu.save_object(score_list, 'forward_feature_score.pkl')
