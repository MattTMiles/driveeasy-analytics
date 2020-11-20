import numpy as np

class EventData:
    def __init__(self, filename=None,
                 timestamp=None,
                 wave1=None,
                 wave2=None,
                 wls1=None,
                 wls2=None,
                 fiber1_id = None,
                 fiber2_id = None,
                 fiber1_sensors = None,
                 fiber2_sensors = None,
                 event_id='event'):
        if filename is not None:
            self._read_event()
        else:
            self.wave1 = wave1
            self.wave2 = wave2,
            self.wls1 = wls1,
            self.wls2 = wls2,
            self.timestamp = timestamp
            self.event_id=event_id

    def load_from_npz(self, filename):
        data = np.load(filename, allow_pickle=True)
        self.timestamp = data['timestamp']
        self.fiber1 = data['fiber1']
        self.fiber2 = data['fiber2']

    def to_dict(self):

        return self.__dict__

    def from_dict(self, data_dict):
        self.__dict__.update(data_dict)

    def _read_event(self, filename):
        if not str(filename).endswith('.npz'):
            raise TypeError('Event filename not ended with .npz')
        data = np.load(filename, allow_pickle=True)
        self.__dict__.update(data)

    def __repr__(self):
        return f'{self.__dict__}'

#%%
import numpy as np
from datetime import datetime, timedelta, timezone
# create sample data
t1 = datetime(2020,11,10,1,10,00)
timestamp = [t1 + timedelta(seconds=i) for i in range(100)]
fiber1 = np.random.random((100, 25)) + 1515.0
fiber2 = np.random.random((100, 25)) + 1519.0
fiber1_id = 3
fiber2_id = 4
fiber1_sensors = list(range(1, 26))
fiber2_sensors = fiber1_sensors
event_id = 'L99E0001'
# save event to npz file
event_filename = 'evt_20201110_011000_L99E0001.npz'
np.savez_compressed(event_filename,
                    timestamp=timestamp,
                    event_id=event_id,
                    wav1=fiber1,
                    wav2=fiber2,
                    fiber1_id=fiber1_id,
                    fiber2_id=fiber2_id,
                    fiber1_sensors=fiber1_sensors,
                    fiber2_sensors=fiber2_sensors)


event_data = {'timestamp':timestamp,
              'event_id':event_id,
              'wav1':fiber1,
              'wav2':fiber2,
              'fiber1_id':fiber1_id,
              'fiber2_id':fiber2_id,
              'fiber1_sensors':fiber1_sensors,
              'fiber2_sensors':fiber2_sensors}
# event_data can then be used in speed extraction algorithms
speed = extract_speed(event_data)

event_data['fiber']
#%% load npz file
event_filename = 'evt_20201110_011000_L99E0001.npz'
event_data = np.load(event_filename)
speed = extract_speed(event_data)
