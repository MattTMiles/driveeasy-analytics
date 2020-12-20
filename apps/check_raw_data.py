#%%
from pydea.io import read_raw_data
from pathlib import Path
#%%
data_dir = Path('../data/M80/333Hz')
file1 = data_dir/'raw_2020-12-12_19-00-24.gzip'
file2 = data_dir/'raw_2020-12-12_19-22-01.gzip'
raw2= read_raw_data(file2)