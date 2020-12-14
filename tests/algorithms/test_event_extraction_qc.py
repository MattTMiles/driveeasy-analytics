import pytest
from pathlib import  Path
from pydea.algorithms.event_extraction import extract_events_qc as EE
import numpy as np
import pandas as pd

def prepare_raw_data():
    data_dir = Path(r'..\test_data')
    file = data_dir/'wav_20201130_223319_F01_UTC.npz'
    data = np.load(file, allow_pickle=True)
    df = pd.DataFrame(index=data['timestamp'], data=data['wav'])
    return df

def test_prepare_data():
    df = prepare_raw_data()
    print(df.head())
    assert df.shape[0] >1

def test_extract_events_qc():
    raw_df = prepare_raw_data()
    events = EE.extract_events(df=raw_df,save_to_file=False)
    print(events)
    assert len(events) > 1

