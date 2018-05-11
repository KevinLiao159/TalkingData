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

# concat train and test
df_concat = pd.concat([df_train[features], df_test[features]])
n_train = df_train.shape[0]
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
            ]
    3. add 'in_test_hh', bool (not sure if helpful)

    4. add [
                'ip_day_test_hh_clicks',
            ]
    5. add counts groupby 3+ features
            [
                'ip_app_device_clicks',
                'ip_app_device_day_clicks',
            ]

    ## X factors below
    6. add item unique counts based on grouper
            [
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
            ]
    7. add mean, var based on grouper
            [
                'ip_app_day_var_hour',
                'ip_device_day_var_hour',
                'ip_os_day_var_hour',

                'ip_channel_day_var_hour',
                'ip_app_os_var_hour',
                'ip_app_channel_var_day',

                'ip_app_channel_mean_hour',
            ]
    8. add cummulative counts based on grouper
            [
                'ip_day_cumcount',
                'ip_cumcount',
                'ip_app_day_cumcount',
                'ip_app_cumcount',
                'ip_device_os_day_cumcount',
                'ip_device_os_cumcount',
            ]
    9. add ['next_click', 'previous_click']
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
others_list = ['app', 'os', 'device']
for col in others_list:
    group = ['ip', col, 'day', 'hour']
    df = df_concat.groupby(group).size().astype('uint16')
    df = pd.DataFrame(df, columns=['ip_{}_day_hour_clicks'.format(col)]).reset_index()    # noqa
    df_train = df_train.merge(df, how='left', on=group)
    df_test_submit = df_test_submit.merge(df, how='left', on=group)


# -----------------------------------------------------------
#  Add feature three: in_test_hh
# -----------------------------------------------------------

# most_freq_hours_in_test_data = list(range(16 + 1)) + [23]
# least_freq_hours_in_test_data = [17, 18, 19, 20, 21, 22]

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
#  Add feature four ip_day_test_hh_clicks
# -----------------------------------------------------------

# ip_day_test_hh_clicks
group = ['ip', 'day', 'in_test_hh']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['ip_day_test_hh_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)


# -----------------------------------------------------------
#  Add feature five: counts groupby 3+ features
# -----------------------------------------------------------


# ip_app_device_clicks
group = ['ip', 'app', 'device']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['ip_app_device_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)

# ip_app_device_day_clicks
group = ['ip', 'app', 'device', 'day']
df = df_concat.groupby(group).size().astype('uint16')
df = pd.DataFrame(df, columns=['ip_app_device_day_clicks']).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)


# -----------------------------------------------------------
#  Add feature six: nunique
# -----------------------------------------------------------

# single 'ip' grouper for ['app', 'device', 'channel']
count_cols = ['app', 'device', 'channel', 'hour']
group = 'ip'
for col in count_cols:
    df = df_concat.groupby(group)[col].nunique().astype('uint16')
    df.name = 'ip_nunique_{}'.format(col)
    df = pd.DataFrame(df).reset_index()
    df_train = df_train.merge(df, how='left', on=group)
    df_test_submit = df_test_submit.merge(df, how='left', on=group)
    #####################
    # self-add
    group_day = ['ip', 'day']
    df = df_concat.groupby(group_day)[col].nunique().astype('uint16')
    df.name = 'ip_day_nunique_{}'.format(col)
    df = pd.DataFrame(df).reset_index()
    df_train = df_train.merge(df, how='left', on=group_day)
    df_test_submit = df_test_submit.merge(df, how='left', on=group_day)
    #####################

# single 'app' grouper for 'channel'
group = 'app'
col = 'channel'
df = df_concat.groupby(group)[col].nunique().astype('uint16')
df.name = 'app_nunique_{}'.format(col)
df = pd.DataFrame(df).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)
#####################
# self-add
group_day = ['app', 'day']
col = 'channel'
df = df_concat.groupby(group_day)[col].nunique().astype('uint16')
df.name = 'app_day_nunique_{}'.format(col)
df = pd.DataFrame(df).reset_index()
df_train = df_train.merge(df, how='left', on=group_day)
df_test_submit = df_test_submit.merge(df, how='left', on=group_day)
#####################

# duble ['ip', 'app'] grouper for 'os'
group = ['ip', 'app']
col = 'os'
df = df_concat.groupby(group)[col].nunique().astype('uint16')
df.name = 'ip_app_nunique_{}'.format(col)
df = pd.DataFrame(df).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)
#####################
# self-add
group_day = ['ip', 'app', 'day']
col = 'os'
df = df_concat.groupby(group_day)[col].nunique().astype('uint16')
df.name = 'ip_app_day_nunique_{}'.format(col)
df = pd.DataFrame(df).reset_index()
df_train = df_train.merge(df, how='left', on=group_day)
df_test_submit = df_test_submit.merge(df, how='left', on=group_day)
#####################

# triple ['ip', 'device', 'os'] grouper for 'app'
group = ['ip', 'device', 'os']
col = 'app'
df = df_concat.groupby(group)[col].nunique().astype('uint16')
df.name = 'ip_device_os_nunique_{}'.format(col)
df = pd.DataFrame(df).reset_index()
df_train = df_train.merge(df, how='left', on=group)
df_test_submit = df_test_submit.merge(df, how='left', on=group)
#####################
# self-add
group_day = ['ip', 'device', 'os'] + ['day']
col = 'app'
df = df_concat.groupby(group_day)[col].nunique().astype('uint16')
df.name = 'ip_device_os_day_nunique_{}'.format(col)
df = pd.DataFrame(df).reset_index()
df_train = df_train.merge(df, how='left', on=group_day)
df_test_submit = df_test_submit.merge(df, how='left', on=group_day)
#####################


# -----------------------------------------------------------
#  Add feature eight: var and mean
# -----------------------------------------------------------

# ip_?_day_var_hour
group_day_cols = ['app', 'device', 'os', 'channel']
col = 'hour'
for group in group_day_cols:
    df = df_concat.groupby(['ip', group, 'day'])[col].var().fillna(0).astype('float32')      # noqa
    df.name = 'ip_{}_day_var_hour'.format(group)
    df = pd.DataFrame(df).reset_index()
    df_train = df_train.merge(df, how='left', on=['ip', group, 'day'])
    df_test_submit = df_test_submit.merge(df, how='left', on=['ip', group, 'day'])           # noqa

# ip_app_os_var_hour
group = 'os'
col = 'hour'
df = df_concat.groupby(['ip', 'app', group])[col].var().fillna(0).astype('float32')          # noqa
df.name = 'ip_app_{}_var_{}'.format(group, col)
df = pd.DataFrame(df).reset_index()
df_train = df_train.merge(df, how='left', on=['ip', 'app', group])
df_test_submit = df_test_submit.merge(df, how='left', on=['ip', 'app', group])

# ip_app_channel_var_day
group = 'channel'
col = 'day'
df = df_concat.groupby(['ip', 'app', group])[col].var().fillna(0).astype('float32')          # noqa
df.name = 'ip_app_{}_var_{}'.format(group, col)
df = pd.DataFrame(df).reset_index()
df_train = df_train.merge(df, how='left', on=['ip', 'app', group])
df_test_submit = df_test_submit.merge(df, how='left', on=['ip', 'app', group])

# ip_app_channel_mean_hour
group = 'channel'
col = 'hour'
df = df_concat.groupby(['ip', 'app', group])[col].mean().fillna(0).astype('float32')         # noqa
df.name = 'ip_app_{}_mean_{}'.format(group, col)
df = pd.DataFrame(df).reset_index()
df_train = df_train.merge(df, how='left', on=['ip', 'app', group])
df_test_submit = df_test_submit.merge(df, how='left', on=['ip', 'app', group])

del df_concat, df
gc.collect()


# -----------------------------------------------------------
#  Add feature nine: cumcount
# -----------------------------------------------------------
# big assumption: ip is encoded different in train set and test set

# ip_cumcount
group = ['ip']
new_col = '{}_cumcount'.format(group[0])
df_train = df_train.sort_values(group + ['click_time'])
df_train[new_col] = df_train.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
df_test_submit = df_test_submit.sort_values(group + ['click_time'])
df_test_submit[new_col] = df_test_submit \
    .groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
#####################
# self-add
# ip_day_cumcount
group = ['ip', 'day']
new_col = '{}_{}_cumcount'.format(group[0], group[1])
df_train = df_train.sort_values(group + ['click_time'])
df_train[new_col] = df_train.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
df_test_submit = df_test_submit.sort_values(group + ['click_time'])
df_test_submit[new_col] = df_test_submit.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
#####################

# ip_app_cumcount
group = ['ip', 'app']
new_col = '{}_{}_cumcount'.format(group[0], group[1])
df_train = df_train.sort_values(group + ['click_time'])
df_train[new_col] = df_train.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
df_test_submit = df_test_submit.sort_values(group + ['click_time'])
df_test_submit[new_col] = \
    df_test_submit.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
#####################
# self-add
# ip_day_cumcount
group = ['ip', 'app', 'day']
new_col = '{}_{}_{}_cumcount'.format(group[0], group[1], group[2])
df_train = df_train.sort_values(group + ['click_time'])
df_train[new_col] = df_train.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
df_test_submit = df_test_submit.sort_values(group + ['click_time'])
df_test_submit[new_col] = df_test_submit.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
#####################

# ip_device_os_cumcount
group = ['ip', 'device', 'os']
new_col = '{}_{}_{}_cumcount'.format(group[0], group[1], group[2])
df_train = df_train.sort_values(group + ['click_time'])
df_train[new_col] = df_train.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
df_test_submit = df_test_submit.sort_values(group + ['click_time'])
df_test_submit[new_col] = df_test_submit.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
#####################
# self-add
# ip_day_cumcount
group = ['ip', 'device', 'os', 'day']
new_col = '{}_{}_{}_{}_cumcount'.format(group[0], group[1], group[2], group[3])
df_train = df_train.sort_values(group + ['click_time'])
df_train[new_col] = df_train.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
df_test_submit = df_test_submit.sort_values(group + ['click_time'])
df_test_submit[new_col] = \
    df_test_submit.groupby(group, sort=True) \
    .apply(lambda x: pd.Series(np.arange(len(x)))) \
    .astype('uint16').values
#####################

# sort index to restore
df_train = df_train.sort_index()
df_test_submit = df_test_submit.sort_index()


# -----------------------------------------------------------
#  Add feature ten: ['next_click', 'previous_click']
# -----------------------------------------------------------

def compute_next_click(_df):
    df = _df[['ip', 'app', 'device', 'os', 'click_time']].copy()
    df['click_time'] = \
        (df['click_time'].astype(np.int64, copy=False) // 10 ** 9).astype(np.int32, copy=False)    # noqa
    return (
        df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - df.click_time    # noqa
        ).astype(np.float32, copy=False)


def compute_previous_click(_df):
    df = _df[['ip', 'app', 'device', 'os', 'click_time']].copy()
    df['click_time'] = \
        (df['click_time'].astype(np.int64, copy=False) // 10 ** 9).astype(np.int32, copy=False)    # noqa
    return (
        df.click_time - df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(1)    # noqa
        ).astype(np.float32, copy=False)


# add next_click
df_train['next_click'] = compute_next_click(df_train)
df_test_submit['next_click'] = compute_next_click(df_test_submit)

# add previous_click
df_train['previous_click'] = compute_previous_click(df_train)
df_test_submit['previous_click'] = compute_previous_click(df_test_submit)

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

    'ip_day_test_hh_clicks',

    'ip_app_device_clicks',
    'ip_app_device_day_clicks',

] + [

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
] + [

    'ip_app_day_var_hour',
    'ip_device_day_var_hour',
    'ip_os_day_var_hour',
    'ip_channel_day_var_hour',

    'ip_app_os_var_hour',
    'ip_app_channel_var_day',
    'ip_app_channel_mean_hour',
] + [
    'ip_day_cumcount',
    'ip_cumcount',
    'ip_app_day_cumcount',
    'ip_app_cumcount',
    'ip_device_os_day_cumcount',
    'ip_device_os_cumcount',
] + [
    'next_click',
    'previous_click',
]

# train
df_train[new_features + ['is_attributed']] \
    .to_hdf('./input/train_v3.hdf', key='train', data_columns=True)
#     .astype('float32') \
del df_train
gc.collect()
# test
df_test_submit[new_features + ['click_id']] \
    .to_hdf('./input/test_v3.hdf', key='test', data_columns=True)
#     .astype('float32') \

del df_test_submit
gc.collect()
t1 = time.time()
t_min = np.round((t1-t0) / 60, 2)
print('It took {} mins to populate new data'.format(t_min))
