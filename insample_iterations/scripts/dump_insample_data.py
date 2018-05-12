import os
import psutil
import time
import numpy as np
import pandas as pd

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
df_all = pd.read_csv(
    filepath_or_buffer="../../data/train.csv",
    usecols=train_columns,
    dtype=dtype,
    low_memory=True,
    parse_dates=['click_time'],
    infer_datetime_format=True,
)
# get training data for insample
# NOTE: below indices are found in EDA
# df_all.iloc[59709852:122070800, :] \
#   .reset_index(drop=True) \
#   .to_hdf('../data/train_raw.hdf', key='foo')
df_all.iloc[:122070800, :] \
    .reset_index(drop=True) \
    .to_hdf('../data/train_raw.hdf', key='train')
# test_supplement
df_all.iloc[122070801:184903442, :] \
    .reset_index(drop=True) \
    .to_hdf('../data/test_supplement_raw.hdf', key='test_supplement')
# test submission
pd.concat(
    [
        df_all.iloc[144708152:152413508, :],
        df_all.iloc[161974465:168265843, :],
        df_all.iloc[174976526:181878211, :]
    ],
    axis=0,
    verify_integrity=True
).reset_index(drop=True).to_hdf('../data/test_raw.hdf', key='test')
t1 = time.time()
t_min = np.round((t1-t0) / 60, 2)
print('It took {} mins to dump in sample data'.format(t_min))
