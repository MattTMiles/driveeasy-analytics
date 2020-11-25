from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(data_files, timestamp_start, timestamp_end):
    df = pd.read_csv(data_files, sep="\t", header=0, index_col=0, low_memory=False).sort_index()
    df.index = pd.to_datetime(df.index, errors='coerce', format='%Y/%m/%d %H:%M:%S:%f')
    df = df[(df.index >= timestamp_start) & (df.index <= timestamp_end)]
    return df


# %% Visualization
def plot_events(lane_data, timestamp_start, timestamp_end, title_name=None):
    fig, ax = plt.subplots()
    plt.scatter(lane_data.index, lane_data["Speed KPH"])
    plt.vlines(lane_data.index, 0, lane_data["Speed KPH"])
    plt.ylabel('Speed (KPH)')
    plt.title(title_name)
    plt.xlim([timestamp_start, timestamp_end])
    ax.set_xticks([])
    plt.legend(['VIPER VIM'])
    plt.xlabel('Timestamps: ' + str(timestamp_start) + ' ~ ' + str(timestamp_end))
    # my_xticks = ax.get_xticks()
    # plt.xticks([my_xticks[0], my_xticks[-1]], visible=True, rotation=00)
    plt.show()


if __name__ == '__main__':
    # note: change below path
    fbgs_data = np.load(
        r'C:\Users\Jin Yan\OneDrive - PARC\FiBridge\Phase_3\DriveEasy\Francis\VIPER VIM validation\1120-1122\driveeasy_wav\melbourne_time_20201120_0900AM\wav_20201119_215814_F04_UTC.npz',
        allow_pickle=True)
    fbgs = pd.DataFrame(data=fbgs_data['wav'],
                        columns=[f'sensor{i + 1}' for i in range(25)],
                        index=fbgs_data['timestamp'])
    fbgs_timestamp = fbgs.index

    # select start & end timestamp
    timestamp_start = fbgs_timestamp[0] + timedelta(hours=10)
    timestamp_end = fbgs_timestamp[-1] + timedelta(hours=10)

    data_files = r'C:\Users\Jin Yan\OneDrive - PARC\FiBridge\Phase_3\DriveEasy\Francis\VIPER VIM validation\1120-1122\43_20201122[1].txt'

    # load data and saparate lanes
    VIM = load_data(data_files, timestamp_start, timestamp_end)
    VIM_east = VIM[VIM['Vehicle Direction'] == 'East']
    VIM_west = VIM[VIM['Vehicle Direction'] == 'West']

    # visualization
    plot_events(VIM_east, timestamp_start, timestamp_end,
                'Vehicle direction: east (Lane 3&4), events count: ' + str(len(VIM_east)))

    plot_events(VIM_west, timestamp_start, timestamp_end,
                'Vehicle direction: west (Lane 1&2), events count: ' + str(len(VIM_west)))
