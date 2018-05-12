import time
import gc
import numpy as np
import pandas as pd
# sklearn imports
from sklearn.metrics.scorer import roc_auc_scorer, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost
# gravity imports
import gravity_learn.utils as gu

# read data
df_train = pd.read_pickle('./input/train_v3.pkl')
# col
target = 'is_attributed'
features = [
    'app', 
    'device', 
    'os', 
    'channel', 
    'dow',
    'doy',
    'hour',
    'minute',
] + [
        '{}_clicks'.format(col) for col in ['ip']
] + [
        '{}_hourly_clicks'.format(col) for col in ['ip', 'app', 'channel']
]
# split
x1, x2, y1, y2 = train_test_split(
    df_train[features], 
    df_train[target], 
    test_size=0.1, 
    random_state=99
)
# prep data
dtrain = xgboost.DMatrix(x1, y1)
dtest = xgboost.DMatrix(x2, y2)
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
# eval_list
watchlist = [(dtrain, 'train'), (dtest, 'valid')]
# train
model = xgboost.train(
    params=params, 
    dtrain=dtrain,
    num_boost_round=200,
    evals=watchlist,
    maximize=True,
    early_stopping_rounds=20, 
    verbose_eval=1
)
####################################################################
t1 = time.time()
t_min = np.round((t1-t0) / 60, 2)
print('It took {} mins to train model'.format(t_min))
# save model
gu.save_object(model, 'model_v3.pkl')

# clean up
del watchlist, dtrain, dtest
gc.collect()

####################################################################
# submit
df_test = pd.read_pickle('./input/test_v3.pkl')
# predict proba
proba = model.predict(xgboost.DMatrix(df_test[features]), ntree_limit=model.best_ntree_limit)
roc_score = roc_auc_score(y_true=df_test[target], y_score=proba)
print('Out of sample roc score is {}'.format(roc_score))
