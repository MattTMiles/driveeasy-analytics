from pathlib import Path
import pandas as pd
import numpy as np

def load_wav_into_dataframe(wav_file):
    data = np.load(wav_file, allow_pickle=True)
    ts = data['timestamp']
    wav = data['wav']
    df = pd.DataFrame(index=ts, data=wav)
    df.columns = [f'sensor{i+1}' for i in range(wav.shape[1])]
    return df


if __name__=='__main__':
    wav_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\Francis\VIPER VIM validation\1120-1122\driveeasy_wav')
    wav_dir = Path(r'C:\Users\matthewmiles\Documents\FiBridge\VIPER VIM validation\driveeasy_data\raw_gzip\wav_20201120_035437_F01_UTC.npz')
    wav_files = list(wav_dir.glob('*.npz'))
    df = load_wav_into_dataframe(wav_files[0])
    print(df.head())