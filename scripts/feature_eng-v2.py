from sys import getsizeof
import time
import numpy as np
import pandas as pd
import gc

# sklearn
from sklearn.model_selection import train_test_split

# gravity imports
import gravity_learn.utils as gu

t0 = time.time()
# spec for train
train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtype = {  
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint16',
    'click_id'      : 'uint32'
}
# read data
df_train = pd.read_csv(
    filepath_or_buffer="./input/train.csv",
    usecols=train_columns,
    dtype=dtype,
    low_memory=True,
    parse_dates=['click_time'],
    infer_datetime_format=True, 
#     skiprows=range(1,122070801),
#     nrows=62832641
)
df_test = pd.read_csv(
    filepath_or_buffer="./input/test_supplement.csv",
    usecols=test_columns,
    dtype=dtype,
    low_memory=True,
    parse_dates=['click_time'],
    infer_datetime_format=True,
#     skiprows=range(1,633),
#     nrows=57537505
)
df_test_submit = pd.read_csv(
    filepath_or_buffer="./input/test.csv",
    usecols=test_columns,
    dtype=dtype,
    low_memory=True,
    parse_dates=['click_time'],
    infer_datetime_format=True, 
)

# set features and targets
features = ['ip', 'app', 'os', 'device', 'channel', 'click_time']
target = 'is_attributed'

# concat train and test
df_concat = pd.concat([df_train[features], df_test[features]])
del df_test
gc.collect()

"""
NOTE: feature engineering ideas
    1. keep 'app', 'os', 'device', 'channel', 'click_time' (hour)
    2. per day per hour count features
            [
                'ip_day_hour_clicks',
                'ip_app_day_hour_clicks',
                'ip_os_day_hour_clicks',
                'ip_device_day_hour_clicks',
                'ip_channel_day_hour_clicks',

                'app_day_hour_clicks',
                'app_os_day_hour_clicks',
                'app_device_day_hour_clicks',
                'app_channel_day_hour_clicks'
            ]
    3. per day count features
            [
                'ip_day_clicks',
                'ip_app_day_clicks',
                'ip_os_day_clicks',
                'ip_device_day_clicks',
                'ip_channel_day_clicks',

                'app_day_clicks',
                'app_os_day_clicks',
                'app_device_day_clicks',
                'app_channel_day_clicks'
            ]

    4. add 'in_test_hh', bool (not sure if helpful)

    5. add [
                'ip_day_test_hh_clicks',
                'app_day_test_hh_clicks',
            ]
    6. add [
                'ip_app_os_day_clicks',
                'ip_app_device_day_clicks',
                'ip_app_os_device_day_clicks',
            ]
"""

# -----------------------------------------------------------
#  Add feature one: add day and hour
# -----------------------------------------------------------

# train
df_train['day'] = df_train['click_time'].dt.day.astype('uint8')
df_train['hour'] = df_train['click_time'].dt.hour.astype('uint8')
# test
df_test_submit['day'] = df_test_submit['click_time'].dt.day.astype('uint8')
df_test_submit['hour'] = df_test_submit['click_time'].dt.hour.astype('uint8')
# concat
df_concat['day'] = df_concat['click_time'].dt.day.astype('uint8')
df_concat['hour'] = df_concat['click_time'].dt.hour.astype('uint8')


# -----------------------------------------------------------
#  Add feature two: per day per hour count features
# -----------------------------------------------------------

# ip_day_hour_clicks
group = ['ip', 'day', 'hour']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['ip_day_hour_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)
# ip binded with 'app', 'os', 'device', 'channel'
others_list = ['app', 'os', 'device', 'channel']
for col in others_list:
    group = ['ip', col, 'day', 'hour']
    df = df_concat.groupby(group).size().astype('uint16')
    df = pd.DataFrame(df, columns=['ip_{}_day_hour_clicks'.format(col)]).reset_index()
    df_train = df_train.merge(df, how='left', on=group)
    df_test_submit = df_test_submit.merge(df, how='left', on=group)

# app_day_hour_clicks
group = ['app', 'day', 'hour']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['app_day_hour_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)
# app binded with 'os', 'device', 'channel'
others_list = ['os', 'device', 'channel']
for col in others_list:
    group = ['app', col, 'day', 'hour']
    df = df_concat.groupby(group).size().astype('uint16')
    df = pd.DataFrame(df, columns=['app_{}_day_hour_clicks'.format(col)]).reset_index()
    df_train = df_train.merge(df, how='left', on=group)
    df_test_submit = df_test_submit.merge(df, how='left', on=group)

# -----------------------------------------------------------
#  Add feature three: per day count features
# -----------------------------------------------------------

# ip_day_clicks
group = ['ip', 'day']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['ip_day_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)
# ip binded with 'app', 'os', 'device', 'channel'
others_list = ['app', 'os', 'device', 'channel']
for col in others_list:
    group = ['ip', col, 'day']
    df = df_concat.groupby(group).size().astype('uint16')
    df = pd.DataFrame(df, columns=['ip_{}_day_clicks'.format(col)]).reset_index()
    df_train = df_train.merge(df, how='left', on=group)
    df_test_submit = df_test_submit.merge(df, how='left', on=group)

# app_day_clicks
group = ['app', 'day']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['app_day_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)
# app binded with 'os', 'device', 'channel'
others_list = ['os', 'device', 'channel']
for col in others_list:
    group = ['app', col, 'day']
    df = df_concat.groupby(group).size().astype('uint16')
    df = pd.DataFrame(df, columns=['app_{}_day_clicks'.format(col)]).reset_index()
    df_train = df_train.merge(df, how='left', on=group)
    df_test_submit = df_test_submit.merge(df, how='left', on=group)

# -----------------------------------------------------------
#  Add feature four: in_test_hh
# -----------------------------------------------------------

most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]

df_train['in_test_hh'] = (
    3 - 2 * df_train.hour.isin(most_freq_hours_in_test_data)
      - 1 * df_train.hour.isin(least_freq_hours_in_test_data)
).astype('uint8')

df_test_submit['in_test_hh'] = (
    3 - 2 * df_test_submit.hour.isin(most_freq_hours_in_test_data)
      - 1 * df_test_submit.hour.isin(least_freq_hours_in_test_data)
).astype('uint8')

df_concat['in_test_hh'] = (
    3 - 2 * df_concat.hour.isin(most_freq_hours_in_test_data)
      - 1 * df_concat.hour.isin(least_freq_hours_in_test_data)
).astype('uint8')

# -----------------------------------------------------------
#  Add feature five ['ip_day_test_hh_clicks', 'app_day_test_hh_clicks']
# -----------------------------------------------------------

# ip_day_test_hh_clicks
group = ['ip', 'day', 'in_test_hh']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['ip_day_test_hh_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)

# app_day_test_hh_clicks
group = ['app', 'day', 'in_test_hh']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['app_day_test_hh_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)

# -----------------------------------------------------------
#  Add feature six
# -----------------------------------------------------------

# ip_app_os_day_clicks
group = ['ip', 'app', 'os', 'day']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['ip_app_os_day_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)

# ip_app_device_day_clicks
group = ['ip', 'app', 'device', 'day']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['ip_app_device_day_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)

# ip_app_os_device_day_clicks
group = ['ip', 'app', 'os', 'device', 'day']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['ip_app_os_device_day_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)


del df_concat
gc.collect()

# -----------------------------------------------------------
#  Save new train / test
# -----------------------------------------------------------

# new_features
new_features = [
    # 'ip',
    'app',
    'device',
    'os',
    'channel',
    'hour',
    'in_test_hh',
] + [
    'ip_day_hour_clicks',
    'ip_app_day_hour_clicks',
    'ip_os_day_hour_clicks',
    'ip_device_day_hour_clicks',
    'ip_channel_day_hour_clicks',
    'app_day_hour_clicks',
    # 'app_os_day_hour_clicks',
    'app_device_day_hour_clicks',
    'app_channel_day_hour_clicks',
] + [
    # 'ip_day_clicks',
    'ip_app_day_clicks',
    'ip_os_day_clicks',
    'ip_device_day_clicks',
    # 'ip_channel_day_clicks',
    # 'app_day_clicks',
    'app_os_day_clicks',
    'app_device_day_clicks',
    # 'app_channel_day_clicks',
] + [
    # 'ip_app_os_day_clicks',
    'ip_app_device_day_clicks',
    'ip_app_os_device_day_clicks',
] + [
    'ip_day_test_hh_clicks',
    # 'app_day_test_hh_clicks',
]

# train
df_train[new_features + ['is_attributed']].to_pickle('./input/train_v5.pkl')

del df_train
gc.collect()
# test
df_test_submit[new_features + ['click_id']].to_pickle('./input/test_v5.pkl')
del df_test_submit
gc.collect()
t1 = time.time()
t_min = np.round((t1-t0) / 60, 2)
print('It took {} mins to populate new data'.format(t_min))
