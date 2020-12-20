from datetime import timedelta, datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as md
import numpy as np
import pandas as pd

class TirtlData:
    def __init__(self, filename,timestamp_start=None, timestamp_end=None):
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.dataframe = self.read_file(filename=filename,
                                        timestamp_start=timestamp_start,
                                        timestamp_end=timestamp_end)

    def read_file(self, filename, timestamp_start=None, timestamp_end=None):
        df = pd.read_csv(filename, sep=',', header=0, parse_dates={'datetime': [0, 1]}, low_memory=False)
        df.index = pd.to_datetime(df['datetime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
        if timestamp_start is not None:
            df = df[(df.index >= timestamp_start) & (df.index <= timestamp_end)]
        return df

    def plot_events(self, figure_title='TIRTL Events'):
        lane_data = self.dataframe
        fig, ax = plt.subplots()
        plt.scatter(lane_data.index, lane_data["speed(kph)"])
        plt.vlines(lane_data.index, 0, lane_data["speed(kph)"])
        plt.ylabel('Speed (KPH)')
        plt.title(figure_title)
        if self.timestamp_start is not None:
            plt.xlim([self.timestamp_start, self.timestamp_end])
        xfmt = md.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        plt.show()



# def load_data(data_files, timestamp_start=None, timestamp_end=None):
#     df = pd.read_csv(data_files,sep=',', header=0, parse_dates={'datetime':[0,1]}, low_memory=False)
#     df.index = pd.to_datetime(df['datetime'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
#     if timestamp_start is not None:
#         df = df[(df.index >= timestamp_start) & (df.index <= timestamp_end)]
#     return df
#
# #%% Visualization
# def plot_events(lane_data, timestamp_start=None, timestamp_end=None, title_name=None):
#     fig, ax = plt.subplots()
#     plt.scatter(lane_data.index, lane_data["speed(kph)"])
#     plt.vlines(lane_data.index, 0, lane_data["speed(kph)"])
#     plt.ylabel('Speed (KPH)')
#     plt.title(title_name)
#     if timestamp_start is not None:
#         plt.xlim([timestamp_start, timestamp_end])
#     xfmt = md.DateFormatter('%H:%M:%S')
#     ax.xaxis.set_major_formatter(xfmt)
#     plt.show()


if __name__ == '__main__':
    # note: change below path
    # fbgs_data = np.load(r'C:\Users\Jin Yan\OneDrive - PARC\FiBridge\Phase_3\DriveEasy\Calibration Test 20201201\M80\M80_DriveEasy\wav\wav_20201202_024632_F01_UTC.npz', allow_pickle=True)
    # fbgs = pd.DataFrame(data=fbgs_data['wav'],
    #                   columns=[f'sensor{i + 1}' for i in range(25)],
    #                   index=fbgs_data['timestamp'])
    # fbgs_timestamp = fbgs.index
    #
    # # select start & end timestamp
    # timestamp_start = fbgs_timestamp[0] + timedelta(hours=11, seconds=3.2)
    # timestamp_end = fbgs_timestamp[-1] + timedelta(hours=11, seconds=3.2)
    
    data_files = r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\Calibration Test 20201201\M80\M80_TIRTL\data-1606945900265_reformat.csv'
    tirtl = TirtlData(data_files)
    # load data and saparate lanes
    # title = load_data(data_files, timestamp_start, timestamp_end)
    # for i in np.arange(1,6):
    #     title_lane = title[title['lane']==i]
    #     # visualization
    #     plot_events(title_lane,
    #                 title_name='Lane '+str(i)+', events count: '+str(len(title_lane)))
    #
