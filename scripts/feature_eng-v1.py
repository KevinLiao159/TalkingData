import os
import psutil
import time
import numpy as np
import pandas as pd
import gc

# memory status
process = psutil.Process(os.getpid())
memused = process.memory_info().rss
print('Total memory in use before reading data: {:.02f} GB '
      ''.format(memused / (2 ** 30)))

t0 = time.time()
# spec for train
train_columns = \
    ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns = \
    ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtype = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint16',
    'click_id': 'uint32'
}
# read data
df_train = pd.read_csv(
    filepath_or_buffer="../data/train.csv",
    usecols=train_columns,
    dtype=dtype,
    low_memory=True,
    parse_dates=['click_time'],
    infer_datetime_format=True,
    # skiprows=range(1, 122070801),
    # nrows=62832641
)
df_test = pd.read_csv(
    filepath_or_buffer="../data/test_supplement.csv",
    usecols=test_columns,
    dtype=dtype,
    low_memory=True,
    parse_dates=['click_time'],
    infer_datetime_format=True,
    # skiprows=range(1, 633),
    # nrows=57537505
)
df_test_submit = pd.read_csv(
    filepath_or_buffer="../data/test.csv",
    usecols=test_columns,
    dtype=dtype,
    low_memory=True,
    parse_dates=['click_time'],
    infer_datetime_format=True,
)
# memory status
memused = process.memory_info().rss
print('Total memory in use after reading data: {:.02f} GB '
      ''.format(memused / (2 ** 30)))

# set features and targets
features = ['ip', 'app', 'os', 'device', 'channel', 'click_time']
target = 'is_attributed'

"""
NOTE:
new_features = [
    'hour',
    'dow',
    'doy',
    'ip_clicks',
    'app_clicks',
    'os_clicks',
    'device_clicks',
    'app_device_comb_clicks',
    'ip_app_comb_clicks',
    'app_os_comb_clicks',
]
"""

# -----------------------------------------------------------
#  Add feature one: add day of week, day of year
# -----------------------------------------------------------

# train
df_train['dow'] = df_train['click_time'].dt.dayofweek.astype('uint16')
df_train['doy'] = df_train['click_time'].dt.dayofyear.astype('uint16')
df_train['hour'] = df_train['click_time'].dt.hour.astype('uint16')
# test
df_test_submit['dow'] = df_test_submit['click_time'].dt.dayofweek.astype('uint16')      # noqa
df_test_submit['doy'] = df_test_submit['click_time'].dt.dayofyear.astype('uint16')      # noqa
df_test_submit['hour'] = df_test_submit['click_time'].dt.hour.astype('uint16')

# -----------------------------------------------------------
#  Add feature two: counts for features per day
# -----------------------------------------------------------

# concat train and test
df_concat = pd.concat([df_train[features], df_test[features]])
del df_test
gc.collect()
# declare features for clicks
features_clicks = ['ip', 'app', 'os', 'device']

for col in features_clicks:
    col_count_dict = dict(df_concat[[col]].groupby(col).size().sort_index())
    # train
    df_train['{}_clicks'.format(col)] = df_train[col].map(col_count_dict).astype('uint16')              # noqa
    # test
    df_test_submit['{}_clicks'.format(col)] = df_test_submit[col].map(col_count_dict).astype('uint16')  # noqa

# -----------------------------------------------------------
#  Add feature three: counts for features per hour
# -----------------------------------------------------------

# features_comb_list = list(itertools.combinations(features_clicks, 2))
features_comb_list = [('app', 'device'), ('ip', 'app'), ('app', 'os')]
for (col_a, col_b) in features_comb_list:
    df = df_concat.groupby([col_a, col_b]).size().astype('uint16')
    df = pd.DataFrame(df, columns=['{}_{}_comb_clicks'.format(col_a, col_b)]).reset_index()             # noqa
    # train
    df_train = df_train.merge(df, how='left', on=[col_a, col_b])
    # test
    df_test_submit = df_test_submit.merge(df, how='left', on=[col_a, col_b])

del df_concat
gc.collect()
# -----------------------------------------------------------
#  Save new train / test
# -----------------------------------------------------------

# new_features
new_features = [
    'app',
    'device',
    'os',
    'channel',
    'hour',
    'dow',
    'doy',
    'ip_clicks',
    'app_clicks',
    'os_clicks',
    'device_clicks',
    'app_device_comb_clicks',
    'ip_app_comb_clicks',
    'app_os_comb_clicks',
]

# train
df_train[new_features + ['is_attributed']].to_pickle('./input/train_v4.pkl')

del df_train
gc.collect()
# test
df_test_submit[new_features + ['click_id']].to_pickle('./input/test_v4.pkl')
del df_test_submit
gc.collect()
t1 = time.time()
t_min = np.round((t1-t0) / 60, 2)
print('It took {} mins to populate new data'.format(t_min))
