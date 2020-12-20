#%%
from pathlib import Path
import numpy as np
import pandas as pd


from pydea.datamodels.datamodel import Event
from pydea.algorithms.speed_estimation.calculate_speed import calculate_speed_alg1, calculate_speed_alg2, calculate_speed_ensemble
#%%
def prepare_event_data(event_dir = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\data\FrancisSt')):
    print(event_dir.exists())
    file = 'extracted_Francis_1201_09_10_lane3_cleaned_event.npz'
    event_file = event_dir/file
    print(f'event file:{event_file}')
    events = np.load(event_file, allow_pickle=True)
    return events
#%%
events = prepare_event_data()
for event in events['events'][0:2]:
    event1 = Event()
    event1.fiber1_id = event.fiber1_id
    event1.fiber2_id = event.fiber2_id
    event1.fiber1_sensors = event.fiber1_sensors
    event1.fiber2_sensors = event.fiber2_sensors
    event1.wav1 = event.wav1
    event1.wav2 = event.wav2
    event1.info = event.info
    event1.timestamp = event.timestamp
    t0 = event.timestamp
    time_str = t0.strftime('%Y%m%d_%H%M%S')
    filename = f'evt_{time_str}_UTC_Francis.npz'
    event1.save_to_file(filename)
    speed_list = calculate_speed_ensemble(event,lane_sensor=[0,1,2,3,4,5,6,7,8,9,20])
    print(speed_list)


