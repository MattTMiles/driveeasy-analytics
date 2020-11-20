#%%
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta, timezone
from pydea.utils import time_utils
#%%
def convert_npz_to_wav_format(data_file, out_dir):
    fiber_id = data_file.stem[-1]
    data = np.load(data_file, allow_pickle=True)
    ts = data['timestamp']
    wav = data['wavelength']
    ln = data['linenumber']
    new_ts = time_utils.interpolate_time_index(ts,ln,time_shift=timedelta(seconds=0))
    t_start = str(ts[0])
    t_start = datetime.strptime(t_start, '%Y/%m/%d %H:%M:%S')
    t_start_str = datetime.strftime(t_start, '%Y%m%d_%H%M%S')
    filename = f'wav_{t_start_str}_F{fiber_id}.npz'
    sensors = list(range(1, wav.shape[1]+1))
    np.savez_compressed(out_dir/filename,
                        timestamp=new_ts,
                        wav=wav,
                        fiber_id=fiber_id,
                        fiber_sensors=sensors,
                        info='Francis St. timestamp is in FBGS time.'
                        )

def convert_dir_to_wav_format(data_dir):
    # data_dir = Path(
    #     r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\Francis\raw_data\10_mins_batches\from_batch64')
    data_files = list(data_dir.glob('*.npz'))
    out_dir = Path(r'wav')
    for data_file in data_files:
        convert_npz_to_wav_format(data_file, out_dir)

def test_wav_data():
    out_dir = Path('wav')
    data = None
    data = np.load(out_dir/'wav_20201107_165626_F1.npz', allow_pickle=True)
    print(data['timestamp'])
    print(data['wav'][0:10])
    assert data is not None
    assert len(data.files)>=4

if __name__=='__main__':
    data_dir = Path(
        r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\Francis\raw_data\10_mins_batches\from_batch64')
    # convert_dir_to_wav_format(data_dir)

    test_wav_data()
    




