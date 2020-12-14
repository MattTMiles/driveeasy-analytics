from collections import namedtuple
import datetime
from loguru import logger
import numpy as np
import pandas as pd


from typing import Union, List, Dict
LaneSensors = namedtuple('LaneSensors', 'line1 line2')

class LaneSensors:
    def __init__(self, leading_sensors, trailing_sensors):
        self.leading_sensors = leading_sensors
        self.trailing_sensors = trailing_sensors

class Raw:
    def __init__(self, filename, use_pandas=True, use_parser=False):
        pass

class RawOperations:

    def extract_events(self, raw: Raw, pick_algorithm:str=None):
        if pick_algorithm == 'qc_v1':
            pass
        elif pick_algorithm == 'hy_v1':
            pass

        elif pick_algorithm == 'yj_v1':
            pass

        else:
            logger.warning('Algorithm not found. Choose from one of these algorithms:{self.algo_list}')

        pass

    def set_annotations(self):
        pass


    def plot_agg_history(self, head=1000):
        """
        Aggregated data column-wise, resulting data index is timestamp. Plot aggregate data.
        Returns:

        """

    def plot_agg_profile(self, head=1000):
        """
        Aggregated data row-wise, resulting data index is sensor id. Plot aggregate data.
        Args:
            head: only plot first few data point specified by `head`.

        Returns:

        """

class LaneConfiguration:
    def __init__(self,
                 lane_id: Union[str,int],
                 fiber_distance:float,
                 leading_fiber_id:int,
                 trailing_fiber_id:int,
                 leading_sensors:List[int],
                 trailing_sensors:List[int]):
        self.fiber_distance = fiber_distance
        self.lane_id = lane_id
        self.leading_fiber_id = leading_fiber_id
        self.trailing_fiber_id = trailing_fiber_id
        self.leading_sensors = leading_sensors
        self.trailing_sensors = trailing_sensors

class RoadConfiguration:
    def __init__(self, raod_id:str, num_lanes:int, lane_configurations:Dict):
        """
        Road configuration
        Args:
            road_id: road name
            num_lanes: number of lanes
            lane_configurations: Dict. Key: lane_id, value: LaneConfiguration
        """
        self.num_lanes = num_lanes
        self.lane_configuration = lane_configurations

class Event:
    def __init__(self, filename=None):
        self.timestamp = datetime.datetime.now()
        self.event_id = 0
        self.fiber1_id = 0
        self.fiber2_id = 0
        self.fiber1_sensors = []
        self.fiber2_sensors = []
        self.info = ''
        self.wav1 = []
        self.wav2 = []
        if filename is not None:
            self._read_file(filename)

    def _read_file(self, filename):
        if not str(filename).endswith('.npz'):
            print('file format not correct. Only .npz files accepted.')
        else:
            event = np.load(filename, allow_pickle=True)
            self.event_id = event['event_id']
            self.fiber1_id = event['fiber1_id']
            self.fiber2_id = event['fiber2_id']
            self.fiber1_sensors = event['fiber1_sensors']
            self.fiber2_sensors = event['fiber2_sensors']
            self.wav1 = event['wav1']
            self.wav2 = event['wav2']
            self.info = event['info']

    def save_to_file(self, filename):
        if self.wav1 == []:
            print('Data empty.')
        else:
            np.savez_compressed(filename,
                                event_id = self.event_id,
                                fiber1_id = self.fiber1_id,
                                fiber2_id = self.fiber2_id,
                                fiber1_sensors = self.fiber1_sensors,
                                fiber2_sensors = self.fiber1_sensors,
                                wav1 = self.wav1,
                                wav2 = self.wav2,
                                info = self.info)

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

