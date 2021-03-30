from scipy import signal
import numpy as np


FIBER_DISTANCE = 2.5
SAMPLING_RATE = 200.0
class SpeedEstimationJY:
    # def __init__(self, fiber_distance=FIBER_DISTANCE, sampling_rate=SAMPLING_RATE):
    #     self.fiber_distance = fiber_distance
    #     self.sampling_rate = sampling_rate
    @staticmethod
    def speed_from_all_sensor_pairs(fiber1, fiber2, low_threshold, high_threshold):
        """Adapted from Jin's implementation.

        :param fiber1:
        :param fiber2:
        :return:
        """
        time_diff = fiber1.idxmax().values - fiber2.idxmax().values
        time_diff = time_diff / np.timedelta64(1, 's')

        # distance (2m) divide by the time difference, 2.23694 is tranferring m/s to mph
        velocity = FIBER_DISTANCE  / time_diff * 3.6

        # threshold the abnormal data beyond vehicle speed limit.
        velocity[(-low_threshold < velocity) & (velocity < low_threshold)] = np.nan
        velocity[(velocity > high_threshold) | (velocity < -high_threshold)] = np.nan
        velocity[velocity > 0] = np.nan
        # mode = histogram_mode(velocity, visualization=False)
        speed_kph  = np.nanmedian(velocity)
        # speed_kph = speed_mps * 3.6
        return speed_kph


class SpeedEstimationQC:
    @staticmethod
    def speed_from_single_sensor_pair(event, sensor_id = 19):
        """Speed estimation using only on pair of sensors from leading and trailing fibers. Adapted from Qiushu's implementation.

        :param event:
        :param sensor_id: sensor to be used for speed estimation
        :return: speed in km/h
        """
        fiber1 = event.fiber1
        fiber2 = event.fiber2
        fiber1_remove_base = fiber1[:,sensor_id-1] -  fiber1[0,sensor_id-1]
        fiber2_remove_base = fiber2[:, sensor_id - 1] - fiber2[0, sensor_id - 1]
        corr = signal.correlate(fiber1_remove_base, fiber2_remove_base)
        speed_mps = (-1 * FIBER_DISTANCE)/(corr.argmax() - len(fiber1))*SAMPLING_RATE
        speed_kph = speed_mps*3.6
        return speed_kph


class SpeedEstimationHY:
    pass




# speed_valid = signal.correlate(event_list[j].data['leading'].iloc[:,ll]-event_list[j].data['leading'].iloc[0,ll], event_list[j].data['trailing'].iloc[:,ll]-event_list[j].data['trailing'].iloc[0,ll])
# speed = -1*2.5/(speed_valid.argmax()-len(event_list[j].data['leading']))*SAMPLING_RATE*3.6
# print(speed)
# plt.figure()
# plt.plot(speed_valid)


class SpeedEstimation_algs:
    #These are the assorted speed algorithms from the ipynb's

    def calculate_speed_qc_alg1(event, lane_sensor):
    
        sos = signal.butter(1, 5, 'hp', fs=SAMPLING_RATE, output='sos')
        filtered_wav1 = signal.sosfilt(sos, event.wav1, axis=0)
        filtered_wav2 = signal.sosfilt(sos, event.wav2, axis=0)
        
        trace_temp1 = np.max(np.abs(filtered_wav1[:, lane_sensor]), axis=1)
        trace_temp2 = np.max(np.abs(filtered_wav2[:, lane_sensor]), axis=1)

        scaled_values_1 = normalize_maxmin(trace_temp1)
        scaled_values_2 = normalize_maxmin(trace_temp2)
        
        speed_valid = signal.correlate(scaled_values_1, scaled_values_2)
        
        peaks_temp2 = signal.find_peaks(scaled_values_2, prominence=0.1)
        peaks_temp1 = signal.find_peaks(scaled_values_1, prominence=0.1)
        if len(peaks_temp1[0]) == 0 or len(peaks_temp2[0]) == 0:
            speed_peak = 0
        else:
            speed_peak = -1*FIBER_DISTANCE / (peaks_temp1[0][0] - peaks_temp2[0][0]) * SAMPLING_RATE * 3.6
        
        if speed_valid.argmax() - len(event.wav1)!= 0:
            speed_corr = -1 * FIBER_DISTANCE / (speed_valid.argmax() - len(event.wav1)) * SAMPLING_RATE * 3.6
        else:
            speed_corr = 0
        return speed_corr, speed_peak

    def calculate_speed_qc_alg2(event, lane_sensor):
        kk = np.min(event.wav1[:, lane_sensor], axis=0).argmin()
        # trace_temp1 = np.max(np.abs(event_list[j].wav1[:,lane_sensor_1]), axis=1)
        trace_temp1 = np.abs(event.wav1[:,kk])
        trace_temp2 = np.abs(event.wav2[:,kk])
        scaled_values_1 = normalize_maxmin(trace_temp1)
        scaled_values_2 = normalize_maxmin(trace_temp2)
    #     trace_temp2 = np.max(np.abs(event.wav2[:, lane_sensor]), axis=1)
    #     trace_temp1 = np.max(np.abs(event.wav1[:, lane_sensor]), axis=1)
        speed_valid = signal.correlate(scaled_values_1, scaled_values_2)
        if speed_valid.argmax() - len(event.wav1)!= 0:
            speed_corr = -1 * FIBER_DISTANCE / (speed_valid.argmax() - len(event.wav1)) * SAMPLING_RATE * 3.6
        else:
            speed_corr = 0

        peaks_temp2 = signal.find_peaks(scaled_values_2, prominence=0.1)
        peaks_temp1 = signal.find_peaks(scaled_values_1, prominence=0.1)
        if len(peaks_temp1[0]) == 0 or len(peaks_temp2[0]) == 0:
            return speed_corr, 0
        else:
            speed_agg_peak = -1*FIBER_DISTANCE / (peaks_temp1[0][0] - peaks_temp2[0][0]) * SAMPLING_RATE * 3.6
            return speed_corr, speed_agg_peak
