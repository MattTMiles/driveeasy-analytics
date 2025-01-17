import numpy as np
import scipy.signal as signal
import datetime

SAMPLING_RATE = 200
FIBER_DISTANCE = 2.5


class Event:
    def __init__(self):
        self.timestamp = datetime.datetime.now()
        self.event_id = 0
        self.fiber1_id = 0
        self.fiber2_id = 0
        self.fiber1_sensors = []
        self.fiber2_sensors = []
        self.info = ''
        self.wav1 = []
        self.wav2 = []


class EventFeatures:
    def __init__(self):
        self.timestamp = datetime.datetime.now()
        self.event_id = 0
        self.fiber1_id = 0
        self.fiber2_id = 0
        self.info = ''
        self.speed = 0
        self.axles = 0
        self.axle_lengths = []
        self.group = 0
        self.class_id = 0


def find_axle_location(wav1, wav2, promin_1=0.001, promin_2=0.001):
    peaks_1 = signal.find_peaks(wav1, prominence=promin_1)
    peaks_2 = signal.find_peaks(wav2, prominence=promin_2)
    axle_count = 0
    axle_list = []
    if len(peaks_1[0]) + len(peaks_2[0]) == 0:
        print('No peaks found!')
        return axle_count, axle_list

    if len(peaks_1[0]) == len(peaks_2[0]):
        axle_list = np.zeros(len(peaks_1))
        axle_count = len(peaks_1)
        shift = peaks_2[0][0] - peaks_1[0][0]
        axle_fiber1 = np.asarray(peaks_1[0]) - peaks_1[0][0]
        axle_fiber2 = np.asarray(peaks_2[0]) - shift - peaks_2[0][0]
        if len(peaks_1[0]) > 1:
            for i in range(1, axle_count):
                try:
                    axle_list[i] = (axle_fiber1[i]+axle_fiber2[i])/2
                except:
                    print('i={}, fiber1={}, fiber2={}'.format(i, len(axle_fiber1), len(axle_fiber2)))
                    axle_list = axle_fiber1
    # Peaks in two channels are different. Choose the channel with more peaks found
    else:
        axle_count = np.max([len(peaks_1[0]), len(peaks_2[0])])
        if axle_count == len(peaks_1[0]):
            axle_list = np.asarray(peaks_1[0]) - peaks_1[0][0]
        else:
            axle_list = np.asarray(peaks_2[0]) - peaks_2[0][0]
    return axle_count, axle_list


def calculate_axle_length(axle_list, speed, sampling_rate):
    axle_length_list = np.zeros(len(axle_list)-1)
    for i in range(len(axle_length_list)):
        axle_length_list[i] = (axle_list[i+1] - axle_list[i])/sampling_rate*speed/3.6
    return axle_length_list


def calculate_speed_qc_alg1(event2, lane_sensor):
    trace_temp1 = np.min(event2.wav1[:, lane_sensor], axis=1)
    trace_temp2 = np.min(event2.wav2[:, lane_sensor], axis=1)

    speed_valid = signal.correlate(trace_temp1, trace_temp2)
    try:
        speed_corr = -1 * 2.5 / (speed_valid.argmax() - len(event2.wav1)) * SAMPLING_RATE * 3.6
    except:
        speed_corr = 0

    return speed_corr


# work with event2 with wls. Use max of absolute value to aggregate data from difference sensors;
def calculate_speed_qc_alg2(event2, lane_sensor):
    kk = np.min(event2.wav1[:, lane_sensor_1], axis=0).argmin()
    # trace_temp1 = np.max(np.abs(event_list[j].wav1[:,lane_sensor_1]), axis=1)
    trace_temp1 = np.abs(event2.wav1[:, kk])
    trace_temp2 = np.abs(event2.wav2[:, kk])

    #     trace_temp2 = np.max(np.abs(event2.wav2[:, lane_sensor]), axis=1)
    #     trace_temp1 = np.max(np.abs(event2.wav1[:, lane_sensor]), axis=1)
    speed_valid = signal.correlate(trace_temp1, trace_temp2)
    try:
        speed_corr = -1 * 2.5 / (speed_valid.argmax() - len(event2.wav1)) * SAMPLING_RATE * 3.6
    except:
        speed_corr = 0
    peaks_temp2 = signal.find_peaks(trace_temp2, prominence=0.002)
    peaks_temp1 = signal.find_peaks(trace_temp1, prominence=0.002)
    if len(peaks_temp1[0]) == 0 or len(peaks_temp2[0]) == 0:
        return speed_corr, 0, 0
    else:
        speed_agg_peak = -1 * 2.5 / (peaks_temp1[0][0] - peaks_temp2[0][0]) * SAMPLING_RATE * 3.6
        return speed_corr, speed_agg_peak, np.max([len(peaks_temp1[0]), len(peaks_temp1[0])])


if __name__ == '__main__':
    # load extracted events
    filename = r'D:\--System--\Matt\Documents\FiBridge\driveeasy-analytics\py_driveeasy\explore\1_qc\data\extracted_Francis_1201_09_10_lane3_cleaned_event.npz'
    events = np.load(filename, allow_pickle=True)
    lane_sensor_1 = np.arange(2, 15)
    event_features_list = []
    print(len(events['events']))
    for i in range(len(events['events'][:20])):
        print(i)
        event_features = EventFeatures()
        axle_length_temp = []
        ax_count = 0
        ax_list = []
        # kk = np.min(events['events'][i].wav1[:, lane_sensor_1], axis=0).argmin()
        # trace_temp1 = np.max(np.abs(event_list[j].wav1[:,lane_sensor_1]), axis=1)
        # trace_temp1 = np.abs(events['events'][i].wav1[:, kk])
        # trace_temp2 = np.abs(events['events'][i].wav2[:, kk])
        trace_temp2 = np.abs(np.min(events['events'][i].wav2[:, lane_sensor_1], axis=1))
        trace_temp1 = np.abs(np.min(events['events'][i].wav1[:, lane_sensor_1], axis=1))
        ax_count, ax_list = find_axle_location(trace_temp1, trace_temp2, promin_1=0.0025, promin_2=0.0025)

        speed_temp1 = calculate_speed_qc_alg1(events['events'][i], lane_sensor_1)
        speed_temp2, speed_temp3, axle_index = calculate_speed_qc_alg2(events['events'][i], lane_sensor_1)
        speed_list = [speed_temp1, speed_temp2, speed_temp3]

        if np.min(speed_list) > 0:
            speed = np.min(speed_list)
        else:
            if np.max(speed_list) < 80:
                speed = np.max(speed_list)
            else:
                speed = np.median(speed_list)

        if ax_count > 1:
            axle_length_temp = calculate_axle_length(ax_list, speed, SAMPLING_RATE)
            if ax_count > 4:
                check_value = 1
                group_check_value = 2
            else:
                check_value = 0.8
                group_check_value = 1.5
            check = axle_length_temp > check_value
            axle_length_temp = axle_length_temp[check]
            ax_count = len(axle_length_temp) + 1
            event_features.group = (axle_length_temp > group_check_value).sum()

        event_features.speed = speed
        event_features.axles = ax_count
        event_features.axle_lengths = axle_length_temp
        event_features_list.append(event_features)

    save_results = False
    if save_results:
        save_filename = filename[:-4] + 'features.npz'
        np.savez_compressed(save_filename, data=event_features_list)


