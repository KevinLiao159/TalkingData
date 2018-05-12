import time
import gc
import numpy as np
import pandas as pd
import itertools
# sklearn imports
from sklearn.metrics.scorer import roc_auc_scorer, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost
# gravity imports
import gravity_learn.utils as gu

# read data
df_train = pd.read_pickle('./input/train_v4.pkl')
df_test = pd.read_pickle('./input/test_v4.pkl')
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
features_clicks = ['ip', 'app', 'os', 'device', 'channel', 'hour']
features_comb_list = list(itertools.combinations(features_clicks, 2))
new_features = [
    'hour',
    'minute',
] + [
        '{}_clicks'.format(col) for col in features_clicks[1:]
] + [
        '{}_{}_comb_clicks'.format(col_a, col_b) for col_a, col_b in features_comb_list
]
score_dict = {}

for feature_add in new_features:
    # prep data
    dtrain = xgboost.DMatrix(df_train[features + [feature_add]], df_train[target])
    print('done data prep!!!')

    t0 = time.time()
    ###################################################################
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
    # clean up
    del dtrain
    gc.collect()
    ####################################################################
    # predict proba
    proba = model.predict(xgboost.DMatrix(df_test[features + [feature_add]]))
    roc_score = roc_auc_score(y_true=df_test[target], y_score=proba)
    print('Out of sample roc score is {}'.format(roc_score))
    score_dict = {**score_dict, **{feature_add: roc_score}}


# save score
gu.save_object(score_dict, 'feature_score_v1.pkl')
