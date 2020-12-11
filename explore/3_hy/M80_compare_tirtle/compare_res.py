#%%
import pandas as pd
from pathlib import Path
large_res = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\m80_1124_res_large.csv')
small_res = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\m80_1124_res_small.csv')
large_df = pd.read_csv(large_res,header=0,index_col='timestamp')#,parse_dates=[0],))
small_df = pd.read_csv(small_res,header=0,index_col='timestamp')#,parse_dates=[0],index_col=0)
print(small_df.head())
