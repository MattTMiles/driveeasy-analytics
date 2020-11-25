from datetime import timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import signal
from collections import namedtuple

from pydriveeasy.event import FeatureExtraction
from pydriveeasy.postprocessing import LaneSensors, LaneData
from pydriveeasy.visualize import plot_heatmap


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for idx, train_row in enumerate(train):
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist, idx))
        distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    neighbors_idx = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
        neighbors_idx.append(distances[i][-1])
    return neighbors, neighbors_idx


def FWHM(arr_x, arr_y):
    difference = max(arr_y) - min(arr_y)
    HM = difference / 2

    pos_extremum = arr_y.argmax()
    if pos_extremum == 0:
        return np.nan
    else:
        nearest_above = (np.abs(arr_y[pos_extremum:-1] - HM)).argmin()
        nearest_below = (np.abs(arr_y[0:pos_extremum] - HM)).argmin()

        FWHM = (np.mean(arr_x[nearest_above + pos_extremum]) -
                np.mean(arr_x[nearest_below]))
        return FWHM


# %% Read data and select starts from 3:45 Melbourne time
df_ch3 = pd.read_pickle('wav_Video3_20201118_0940AM_ch_1.pickle')
df_ch3.index = pd.to_datetime(df_ch3.index)
df_ch3_0 = df_ch3.iloc[0, :]
# df_ch3 = df_ch3.iloc[4400::,:] - df_ch3.iloc[4400,:].values.squeeze()
df_ch3 = df_ch3 - df_ch3.iloc[0, :].values.squeeze()
df_ch3.plot()

df_ch4 = pd.read_pickle('wav_Video3_20201118_0940AM_ch_2.pickle')
df_ch4.index = pd.to_datetime(df_ch4.index)
df_ch4_0 = df_ch4.iloc[0, :]

# df_ch4 = df_ch4.iloc[4400::,:] - df_ch4.iloc[4400,:].values.squeeze()
df_ch4 = df_ch4 - df_ch4.iloc[0, :].values.squeeze()

df_ch4.plot()

# %% Put sensors in lanes
lane1_line1 = df_ch3.iloc[:, 0:10]
lane1_line2 = df_ch4.iloc[:, 0:10]
lane1 = [lane1_line1, lane1_line2]

lane2_line1 = df_ch3.iloc[:, 11:20]
lane2_line2 = df_ch4.iloc[:, 11:20]
lane2 = [lane2_line1, lane2_line2]

lane_left1 = df_ch3.iloc[:, 5:25]
lane_left2 = df_ch4.iloc[:, 5:25]


def histogram_mode(velocity, visualization=True):
    """
    --Input:
    velocity: 25 sensing points velcotiy estimation.
    visulization: choice of visualization or not

    --Output:
    mode: most frequent value

    """
    data = pd.Series(velocity)
    # Make bins
    bins = np.arange(data.min(), data.max() + 20)
    # Compute histogram
    h, _ = np.histogram(data, bins)
    # Find most frequent value
    mode = bins[h.argmax()]

    if visualization == True:
        plt.figure()
        plt.hist(velocity, color='blue', edgecolor='black',
                 bins=bins)
        plt.axvline(mode, color='k', linestyle='dashed', linewidth=1)
        plt.title('Histogram of velocity estimation')
        plt.xlabel('Velocity (mph)')
        plt.ylabel('Frequency')
        plt.show()
    return mode


def speed_estimate(fiber1, fiber2):
    # indentify the index of peak values

    # fwhm = []
    # for i in range(fiber1.shape[1]):
    #     fwhm.append(FWHM(np.arange(len(fiber1)), fiber1.iloc[:,i].values))
    # fwhm = np.asarray(fwhm)

    # outlier = np.argwhere(fwhm > 0.5)
    # fiber1.iloc[:,outlier]=np.nan
    # fiber2.iloc[:,outlier]=np.nan

    # time difference from peak values
    time_diff = fiber1.idxmax().values - fiber2.idxmax().values
    time_diff = time_diff / np.timedelta64(1, 's')

    # distance (2m) divide by the time difference, 2.23694 is tranferring m/s to mph
    velocity = 2.5 / time_diff * 3.6

    # threshold the abnormal data beyond vehicle speed limit.
    velocity[(-10 < velocity) & (velocity < 10)] = np.nan
    velocity[(velocity > 150) | (velocity < -150)] = np.nan
    velocity[velocity > 0] = np.nan
    # mode = histogram_mode(velocity, visualization=False)
    mode = np.nanmedian(velocity)
    return mode


# %% moving window for Lane2:
fs = 200  # Hz, sampling frequency
window_size = int(1 * fs)  # 1 second window size
speed_dic = []
count = 0
for i in range(int(lane_left1.shape[0] / window_size) - 1):
    event = 0

    # prepare the windows for analysis
    window_i1 = lane_left1.iloc[window_size * (i):window_size * (i + 1), :]
    window_i1 = window_i1 - window_i1.iloc[0, :].values.squeeze()
    window_i1.reset_index(drop=True, inplace=True)

    start_index = window_i1.index[0]
    end_index = window_i1.index[-1]

    window_i2 = lane_left1.iloc[window_size * (i):window_size * (i + 1), :]
    window_i2 = window_i2 - window_i2.iloc[0, :].values.squeeze()
    window_i2.reset_index(drop=True, inplace=True)

    # threshold the window see if peaks exist
    # if window_i1[window_i1.abs()*df_ch4_0[5:25].values>2].any().any():
    if window_i1[window_i1.abs() > 0.001].any().any():

        event = 1
        # indentify the maximum response sensor id
        sensor1 = (window_i1.abs().values.argmax() + 1) % 25 - 1
        sensor2 = (window_i2.abs().values.argmax() + 1) % 25 - 1

        if sensor1 | sensor2 > 4:
            sensor1 -= 4
            sensor2 -= 4
            # make sure the line1 and line2 sensor ids are the same from one vehicle.
            if sensor1 == sensor2:
                time_peaks1, peak_prop1 = signal.find_peaks(abs(window_i1.iloc[:, sensor1]), distance=10, height=0.001,
                                                            width=0.5)
                time_peaks2, peak_prop2 = signal.find_peaks(abs(window_i2.iloc[:, sensor2]), distance=10, height=0.001,
                                                            width=0.5)
                n_peaks = len(time_peaks1)

                # make sure the number of peaks are largern than 1
                if n_peaks > 1:
                    # # identify the peaks (for 'signal.find_peaks' parameter adjusting purposes)
                    # # comment out if in use
                    # plt.figure()
                    # window_i1.iloc[:,sensor1].plot()
                    # plt.plot(time_peaks1, window_i1.iloc[time_peaks1,sensor1], "x")
                    # plt.vlines(x=time_peaks1, ymin=window_i1.iloc[time_peaks1,sensor1] - peak_prop1["prominences"],
                    #     ymax = window_i1.iloc[time_peaks1,sensor1], color = "C1")
                    # plt.hlines(y=peak_prop1["width_heights"], xmin=peak_prop1["left_ips"],
                    #             xmax=peak_prop1["right_ips"], color = "C1")
                    # plt.show()

                    # preset window peaks properties
                    window_widths = np.zeros((10, 30))
                    window_left = np.zeros((10, 30))
                    window_right = np.zeros((10, 30))

                    # select adjacent 20 windows for analysis
                    for j in range(10):
                        window_s_start = window_size * (i) - 500 + (100 * j)
                        window_s_end = window_size * (i + 1) - 500 + (100 * j)

                        # smaller sliding windows near the window_i for analysis
                        window_s1 = lane_left1.iloc[window_s_start:window_s_end, :] - lane_left1.iloc[window_s_start,
                                                                                      :].values.squeeze()  # zero the window
                        window_s2 = lane_left2.iloc[window_s_start:window_s_end, :] - lane_left2.iloc[window_s_start,
                                                                                      :].values.squeeze()  # zero the window

                        # get the peaks in this sliding windows
                        time_peaks, peak_prop = signal.find_peaks(abs(window_s1.iloc[:, sensor1]), distance=10,
                                                                  height=0.001, width=0.5)
                        n_peaks_temp = len(time_peaks)
                        window_widths[j, 0:n_peaks_temp] = peak_prop["widths"]
                        window_left[j, :n_peaks_temp] = peak_prop["left_bases"]
                        window_right[j, :n_peaks_temp] = peak_prop["right_bases"]

                        # Make a classification prediction with neighbors
                    distances = list()
                    for idx, train_row in enumerate(window_widths):
                        dist = euclidean_distance(peak_prop1["widths"], train_row)
                        distances.append((train_row, dist, idx))
                        distances.sort(key=lambda tup: tup[1])
                    neighbors = list()
                    neighbors_idx = list()

                    for ii in range(len(distances)):
                        if distances[ii][1] < 1e-3:
                            neighbors.append(distances[ii][0])
                            neighbors_idx.append(distances[ii][-1])

                    event_start, event_end = 0, 0
                    event_window = []

                    if len(neighbors_idx) >= 2:
                        event_left_index = np.argmin(neighbors_idx)
                        event_left_window = neighbors_idx[event_left_index]
                        event_start = window_left[event_left_window, 0]
                        event_start = int(window_size * (i) - 500 + (100 * event_left_window) + event_start - 10)

                        event_right_index = np.argmax(neighbors_idx)
                        event_right_window = neighbors_idx[event_right_index]
                        event_end = window_right[event_right_window, :].max()
                        event_end = int(window_size * (i) - 500 + (100 * event_right_window) + event_end + 10)

                        event_window = lane_left1.iloc[event_start:event_end, :]
                        event_window.index = pd.to_datetime(event_window.index)
                        event = 1
                        # str(timedelta(seconds=event_start))

                        speed = -speed_estimate(lane_left1.iloc[event_start:event_end, :],
                                                lane_left2.iloc[event_start:event_end, :])
                        speed_dic.append([event_start, speed])
                        plot_heatmap(event_window, 'Lane12_Event' + str(count),
                                     str(lane_left1.index[window_size * (i)]) + ', speed (km/h)' + str(speed))
                    else:
                        # timedelta(seconds=400*i))

                        speed = -speed_estimate(lane_left1.iloc[window_size * (i):window_size * (i + 1), :],
                                                lane_left2.iloc[window_size * (i):window_size * (i + 1), :])
                        speed_dic.append([window_size * (i), speed])
                        plot_heatmap(lane_left1.iloc[window_size * (i):window_size * (i + 1), :],
                                     'Lane12_Event' + str(count),
                                     str(lane_left1.index[window_size * (i)]) + ', )' + "%.2f" % speed + 'km/h')
                    count += 1
        else:
            event = 0

fig, ax = plt.subplots()
for i in range(len(speed_dic)):
    plt.vlines(speed_dic[i][0], 0, speed_dic[i][1], color='b')
    plt.scatter(speed_dic[i][0], speed_dic[i][1], color='b')
ax.set_ylabel('speed (km/h)')
ax.set_ylim([0, 160])
ax.set_xticks([])
ax.set_title('event counts: ' + str(i + 1))
plt.show()