from datetime import timedelta, datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(data_files, timestamp_start=None, timestamp_end=None):
    df = pd.read_csv(data_files, sep=",", header=0, index_col=0, low_memory=False)
    if timestamp_start is not None:
        df = df[(df.index.get_level_values(0) >= timestamp_start) & (df.index.get_level_values(0) <= timestamp_end)]    
    return df

#%% Visualization
def plot_events(lane_data, timestamp_start=None, timestamp_end=None, title_name=None):
    fig, ax = plt.subplots()
    plt.scatter(lane_data.time, lane_data["speed(kph)"])
    plt.vlines(lane_data.time, 0, lane_data["speed(kph)"])
    plt.ylabel('Speed (KPH)')
    plt.title(title_name)
    if timestamp_start is not None:
        plt.xlim([timestamp_start, timestamp_end])
    my_xticks = ax.get_xticks()
    plt.xticks([my_xticks[0], my_xticks[-1]], visible=True, rotation=00)
    plt.show()


if __name__ == '__main__':
    # note: change below path
    data_files = r'C:\Users\Jin Yan\OneDrive - PARC\FiBridge\Phase_3\DriveEasy\M80\TIRTLE validation\20201124\TIRTLE\T0490_vehicle_20201125_070000_+1100_1h.csv'
    # select start & end timestamp
    # timestamp_start = '2020/11/20 00:01:13.000'
    # timestamp_end = '2020/11/20 00:12:18'
    
    # load data and saparate lanes
    title = load_data(data_files)    
    for i in np.arange(1,6):
        title_lane = title[title['lane']==str(i)]
        # visualization
        plot_events(title_lane, 
                    title_name='Lane '+str(i)+', events count: '+str(len(title_lane)))

