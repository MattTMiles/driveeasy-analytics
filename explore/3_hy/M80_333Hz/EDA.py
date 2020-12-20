%load_ext autoreload
%autoreload 2
#%%
from pydea.datamodels import Wav
from pathlib import Path
import numpy as np
import pandas as pd
# from pydea.algorithms.event_extraction.extract_events_hy import create_event_data
from pydea.algorithms.event_extraction.extract_events_qc import extract_events
from pydea.algorithms.event_extraction.extract_events_hy import two_stage_event_extraction, create_event_data
from pydea.viz.event import  plot_events

#%%
data_dir = Path(
    r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\HighFrequency333Hz_20201213\HighFreq_333Hz\wav')
wav_file = data_dir / 'wav_20201212_195900_F03_UTC.npz'
wav1 = Wav(wav_file)
wav11 = wav1.clip(pick_range=[0,10000])
wav11.to_wls()

wav_file2 = data_dir / 'wav_20201212_195900_F04_UTC.npz'
wav2 = Wav(wav_file2)
wav22 = wav2.clip(pick_range=[0,10000])
wav22.to_wls()

#%% extract events using Hy_algo1
large_peaks, large_index_range, small_peaks, small_index_range = two_stage_event_extraction(abs(wav11.wls.min(axis=1)), large_threshold=15, small_threshold=2.5)
events = create_event_data(wav11, wav22, large_index_range, event_info='large')
events2 = create_event_data(wav11, wav22, small_index_range, event_info='small')
events.extend(events2)
#%% save events

np.savez_compressed('events_20201212_195900_UTC_M80_333Hz.npz',
                    events = events)
import pickle
with open('events_20201212_195900_UTC_M80_333Hz.pickle', 'wb') as f:
    pickle.dump(events, f)
#%% extract features
from pydea.algorithms.count_axles.count_axles_qc import calculate_axle_number_distance
for event in events:
    axle_count, axle_distance,num_groups = calculate_axle_number_distance(event,lane_sensor=list(range(0,11)), promin_1=0.001, promin_2=0.001,sampling_rate=333)
    print(f'{event.event_id}, num axles: {axle_count}, ax distance:{axle_distance}, num groups:{num_groups}')

#%% plot

wls_df = pd.DataFrame(index=wav11.timestamp, data=wav11.wls.min(axis=1))
#%%
from plotly.offline import plot
import datetime
t1 = datetime.datetime(2020,12,12,19,59,20)
fig = px.line(wls_df)
for event in events:
    start = event.timestamp[0]
    end = event.timestamp[-1]
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    print(f'start:{start}')
    fig.add_vline(x=start_dt,line=dict(color="LightSeaGreen", width=3,))
    fig.add_vline(x=end_dt,line=dict(
                      color="Blue",
                      width=3,))
    # fig.add_hline(y=1)
plot(fig)
    # fig.add_shape(type="line",
    #               xref="x", yref="y",
    #               x0=start, y0=0, x1=start, y1=1,
    #               line=dict(
    #                   color="LightSeaGreen",
    #                   width=3,
    #               ))
    # fig.add_shape(type="line",
    #               xref="x", yref="y",
    #               x0=end, y0=0, x1=end, y1=2,
    #               line=dict(
    #                   color="Blue",
    #                   width=3,
                  # ))
# plot(fig,filename='events_and_wls.html')
#%%
plot_events(events=events, wls_df=wls_df)

#%%
df = pd.DataFrame(index=wav11.timestamp, data=wav11.wls)
df.columns = [f'sensor_{i+1}' for i in range(wav1.wav.shape[1])]
#%%
wav11.plot_agg_history()
#%%
#%%
events = extract_events(df,save_to_file=True,threshold=1.0)
#%%
lane5 = df.iloc[:,list(range(0,11))]
#%%
events = extract_events(lane5,threshold=0.1 )






#%%
np.savez_compressed('wav_20201212_180000_F03_UTC_M80_333Hz.npz',
                    timestamp=wav11.timestamp,
                    wav=wav11.wav,
                    wls=wav11.wls,
                    fiber_id=3,
                    info='M80 333Hz, 5 events.')
#%%
