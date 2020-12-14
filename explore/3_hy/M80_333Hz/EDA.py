from pydea.datamodels import Wav
from pathlib import Path
import numpy as np
import pandas as pd

#%%
data_dir = Path(
    r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\HighFrequency333Hz_20201213\HighFreq_333Hz\wav')
wav_file = data_dir / 'wav_20201212_195900_F03_UTC.npz'
wav1 = Wav(wav_file)
df = pd.DataFrame(index=wav1.timestamp, data=wav1.wav)
df.columns = [f'sensor_{i+1}' for i in range(wav1.wav.shape[1])]

#%%
from pydea.algorithms.event_extraction.extract_events_qc import extract_events

events = extract_events(df,save_to_file=False)