from datetime import timedelta, datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(data_files, timestamp_start, timestamp_end):
    df = pd.read_csv(data_files, sep="\t", header=0, index_col=0, low_memory=False)
    df = df[(df.index.get_level_values(0) >= timestamp_start) & (df.index.get_level_values(0) <= timestamp_end)]
    return df


# %% Visualization
def plot_events(lane_data, timestamp_start, timestamp_end, title_name=None):
    fig, ax = plt.subplots()
    plt.scatter(lane_data.index, lane_data["Speed KPH"])
    plt.vlines(lane_data.index, 0, lane_data["Speed KPH"])
    plt.ylabel('Speed (KPH)')
    plt.title(title_name)
    plt.xlim([timestamp_start, timestamp_end])
    my_xticks = ax.get_xticks()
    plt.xticks([my_xticks[0], my_xticks[-1]], visible=True, rotation=00)
    plt.show()


if __name__ == '__main__':
    # note: change below path
    data_files = r'C:\Users\Jin Yan\OneDrive - PARC\FiBridge\Phase_3\DriveEasy\Francis\VIPER VIM validation\1120-1122\43_20201122[1].txt'
    # select start & end timestamp
    timestamp_start = '2020/11/20 00:01:13.000'
    timestamp_end = '2020/11/20 00:12:18'

    # load data and saparate lanes
    VIM = load_data(data_files, timestamp_start, timestamp_end)
    VIM_east = VIM[VIM['Vehicle Direction'] == 'East']
    VIM_west = VIM[VIM['Vehicle Direction'] == 'West']

    # visualization
    plot_events(VIM_east, timestamp_start, timestamp_end,
                'Vehicle direction: east (Lane 3), events count: ' + str(len(VIM_east)))

    plot_events(VIM_east, timestamp_start, timestamp_end,
                'Vehicle direction: west (Lane 1&2), events count: ' + str(len(VIM_west)))
