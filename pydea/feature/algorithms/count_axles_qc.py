import numpy as np
import scipy.signal as signal
import datetime

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

# look for large slope in the time sequence
def event_detection(data_trace, threshold=0.001, seg_length=3):
    event_flag = []
    for i in range(int(len(data_trace) / seg_length)):
        event_flag.append(define_baseline_alg1(data_trace[i * seg_length:(i + 1) * seg_length], threshold=threshold))
        # event_flag = define_baseline_alg2(data_trace, moving_ave, threshold=0.001)
    return event_flag


def define_baseline_alg1(data_seg, threshold):
    if np.max(np.abs(data_seg - np.median(data_seg))) > threshold:
        return 1
    else:
        return 0


def axle_qc_alg3(event, lane_sensor):
    # pick sensor that produces the min signal on either half of the lane. certain overlap region
    k_1 = np.min(event.wav1[:, lane_sensor[0:7]], axis=0).argmin()
    l_1 = np.min(event.wav1[:, lane_sensor[5:]], axis=0).argmin()
    k_2 = np.min(event.wav2[:, lane_sensor[0:7]], axis=0).argmin()
    l_2 = np.min(event.wav2[:, lane_sensor[5:]], axis=0).argmin()

    trace_temp1 = np.abs(event.wav1[:, lane_sensor[k_1]]) + np.abs \
        (event.wav1[:, lane_sensor[5 + l_1]])
    trace_temp2 = np.abs(event.wav2[:, lane_sensor[k_2]]) + np.abs \
        (event.wav2[:, lane_sensor[5 + l_2]])

    signal_strength = np.max([np.max(trace_temp1), np.max(trace_temp2)])

    if signal_strength > 0.05:
        # threshold 0.003 for Francis; 0.0015 for M80
        filter_threshold = 0.003
    else:
        # threshold 0.0015 for Francis; 0.0008 - 0.001 for M80
        filter_threshold = 0.0015

    axle_filter_1 = event_detection(trace_temp1, threshold=filter_threshold, seg_length=3)
    axle_filter_2 = event_detection(trace_temp2, threshold=filter_threshold, seg_length=3)
    ax_count_1 = signal.find_peaks(axle_filter_1, prominence=0.5, distance=3)
    ax_count_2 = signal.find_peaks(axle_filter_2, prominence=0.5, distance=3)
    ax_count = np.max([len(ax_count_1[0]), len(ax_count_2[0])])

    # if vehicle too light
    if ax_count < 2:
        axle_filter_1 = event_detection(trace_temp1, threshold=0.001, seg_length=3)
        axle_filter_2 = event_detection(trace_temp2, threshold=0.001, seg_length=3)
        ax_count_1 = signal.find_peaks(axle_filter_1, prominence=0.5, distance=3)
        ax_count_2 = signal.find_peaks(axle_filter_2, prominence=0.5, distance=3)
        ax_count = np.min([len(ax_count_1[0]), len(ax_count_2[0])])

    if ax_count >= 2:
        if len(ax_count_1[0]) > len(ax_count_2[0]):
            ax_list = (ax_count_1[1]['left_bases'] + ax_count_1[1]['right_bases']) / 2
        else:
            ax_list = (ax_count_2[1]['left_bases'] + ax_count_2[1]['right_bases']) / 2
    else:
        if len(ax_count_1[0]) < len(ax_count_2[0]):
            ax_list = (ax_count_1[1]['left_bases'] + ax_count_1[1]['right_bases']) / 2
        else:
            ax_list = (ax_count_2[1]['left_bases'] + ax_count_2[1]['right_bases']) / 2

    return ax_count, ax_list

if __name__ == "__main__":
    pass

