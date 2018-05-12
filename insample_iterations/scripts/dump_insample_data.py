from sys import getsizeof
import time
import numpy as np
import pandas as pd
import itertools
import gc

# sklearn
from sklearn.model_selection import train_test_split

# gravity imports
import gravity_learn.utils as gu

t0 = time.time()
# spec for train
train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtype = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}
# load data
df_all = pd.read_csv(
    filepath_or_buffer="../input/train.csv",
    usecols=train_columns,
    dtype=dtype,
    low_memory=True,
    parse_dates=['click_time'],
    infer_datetime_format=True
)
# train
# df_all.iloc[59709852:122070800, :].reset_index(drop=True).to_pickle('./input/train.pkl')
df_all.iloc[:122070800, :].reset_index(drop=True).to_pickle('./input/train.pkl')
# test_supplement
df_all.iloc[122070801:184903442, :].reset_index(drop=True).to_pickle('./input/test_supplement.pkl')
# test submission
pd.concat(
    [
        df_all.iloc[144708152:152413508, :], 
        df_all.iloc[161974465:168265843, :], 
        df_all.iloc[174976526:181878211, :]
    ], 
    axis=0, 
    verify_integrity=True
).reset_index(drop=True).to_pickle('./input/test.pkl')
t1 = time.time()
t_min = np.round((t1-t0) / 60, 2)
print('It took {} mins to dump in sample data'.format(t_min))