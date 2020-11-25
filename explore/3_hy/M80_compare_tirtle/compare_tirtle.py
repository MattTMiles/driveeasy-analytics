from pydea.ETL import tirtle, wav
from pathlib import Path
import pandas as pd
import numpy as np
#%%
tirtle_file = r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\TIRTLE\T0490_vehicle_20201125_070000_+1100_1h.csv'
tirtle_data = tirtle.load_data(tirtle_file)

# data_files = list(data_dir.glob('*.npz'))
#%%
data_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\driveeasy_wav')

ch1_file = 'wav_20201124_195912_F01_UTC.npz'
ch2_file = 'wav_20201124_195912_F02_UTC.npz'
ch3_file = 'wav_20201124_195912_F03_UTC.npz'
ch4_file = 'wav_20201124_195912_F04_UTC.npz'

df1 = wav.load_wav_into_dataframe(data_dir/ch1_file)
#%%
import matplotlib.pyplot as plt
plt.figure()
df1.sensor19.iloc[0:-90000].plot()
plt.show()
