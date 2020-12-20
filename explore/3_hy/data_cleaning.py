#%%
import pandas as pd
import numpy as np
from pydea.ETL.wav import load_wav_into_dataframe
from pathlib import Path
#%%
data_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\raw\raw_gzip\melbourne_time_20201125_0700AM\wav')

ch1_file = 'wav_20201124_195912_F01_UTC.npz'
ch2_file = 'wav_20201124_195912_F02_UTC.npz'
ch3_file = 'wav_20201124_195912_F03_UTC.npz'
ch4_file = 'wav_20201124_195912_F04_UTC.npz'

df1 = load_wav_into_dataframe(data_dir/ch1_file)
#%%
df11 = df1.drop(pd.Timestamp('2020-11-23 17:08:49.377836'))
#%%
ch_file  = ch4_file
df = load_wav_into_dataframe(data_dir/ch_file)
df = df.drop(pd.Timestamp('2020-11-23 17:08:49.377836'))
ts = df.index
wav = df.values
np.savez_compressed(data_dir/ch_file,
                    timestamp=ts,
                    wav=wav)


#%%

ts = df.index
dt = np.diff(ts)
nb = np.sum([t.astype('int')<0 for t in dt])
print(nb)
#%%
for i in range(1,len(ts)):
    if (ts[i] - ts[i-1]).total_seconds()<0:
        print(i,ts[i-1], ts[i])
# found bad raw: 2020-11-23 17:08:49.377836
# for i, t in enumerate(ts):
#     if t
