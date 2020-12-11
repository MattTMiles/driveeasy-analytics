
from scipy import fft, signal
import numpy as np
from sklearn.preprocessing import minmax_scale

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
    def __init__(self,
                 fiber1_data,
                 fiber2_data,
                 fiber_distance=2.5,
                 frequency=200.0):
        self.frequency = frequency
        self.sensor_pairs = []
        self.lane_sensors = {'lane_1': []}
        self.sensor_locations = []
        self.fiber1_data = fiber1_data
        self.fiber2_data = fiber2_data
        self.fiber_distance = fiber_distance

    def estimate_delta_points(self, data1, data2, method='correlation'):
        """method ='correlation', 'fft'
        """
        if method == 'correlation':
            len_signal = len(data1)
            s1 = np.argmax(signal.correlate(data1, data2))
            #     s2 = np.argmax(signal.correlate(data2,data1))
            shift = s1 - len_signal
        elif method == 'fft':
            af = fft.fft(data1)
            bf = fft.fft(data2)
            c = fft.ifft(af * np.conj(bf))
            shift = np.argmax(abs(c))
        return shift

    def integer_to_time(self, number):
        return number / self.frequency

    def estimate_delta_t(self, data1, data2, method='correlation'):
        shift = self.estimate_delta_points(data1, data2, method=method)
        return self.integer_to_time(shift)

    def estimate_speed_2sensors(self, s1, s2):
        # fe = FeatureExtraction(s1, s2, frequency=SAMPLE_FREQUENCY)
        delta_t = self.estimate_delta_t(s1, s2)
        speed = self.fiber_distance / delta_t
        return -speed

from pathlib import Path
import pandas as pd

def test_speed_estimation():
    data_dir = Path(
        r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\extracted_events')
    # unique_events = get_unique_events(data_dir)
    file1 = 'lane5_large_event1_line1.csv'
    file2 = 'lane5_large_event1_line2.csv'
    line1 = pd.read_csv(data_dir/file1)
    line2 = pd.read_csv(data_dir/file2)
    SE = SpeedEstimationHY(line1, line2)
    s = SE.estimate_speed_2sensors(line1.min(axis=1), line2.min(axis=1))
    # s = -s
    print(f'speed: {s} m/s or {s*3.6:.3f} kmh')
    assert s>0 and s<200/3.6







# speed_valid = signal.correlate(event_list[j].data['leading'].iloc[:,ll]-event_list[j].data['leading'].iloc[0,ll], event_list[j].data['trailing'].iloc[:,ll]-event_list[j].data['trailing'].iloc[0,ll])
# speed = -1*2.5/(speed_valid.argmax()-len(event_list[j].data['leading']))*SAMPLING_RATE*3.6
# print(speed)
# plt.figure()
# plt.plot(speed_valid)
