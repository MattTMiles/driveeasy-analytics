from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from loguru import logger
from plotly.offline import plot

from pydea.preprocessing.detrend import detrend_df
from pydea.preprocessing.remove_outliers import remove_outliers_df

class Wav:
    def __init__(self, filename=None):
        self.wav = None
        self.wls = None
        self.fiber_id = None
        if filename is not None:
            self.filename = filename
            self._read_file(filename)

    def _read_file(self, filename):
        if not str(filename).endswith('.npz'):
            print('file format not correct. Only .npz files accepted.')
        else:
            data = np.load(filename, allow_pickle=True)
            self.timestamp = data['timestamp']
            self.wav = data['wav']
            self.fiber_id = data['fiber_id']

    def save_to_file(self, filename=None):
        if self.wav == []:
            print('Wav data empty; unable to save to file.')
        else:
            if filename is None:
                ts = pd.Timestamp(self.timestamp[0])
                time_str = ts.strftime('%Y%m%d_%H%M%S')
                filename = f'wav_{time_str}_F{self.fiber_id}.npz'
            np.savez_compressed(filename,
                                timestamp=self.timestamp,
                                wav=self.wav,
                                fiber_id=self.fiber_id)

    def clip(self, pick_range=[0, 10000]):
        new_wav = Wav()
        start, end = pick_range
        end = min(len(self.timestamp), end)
        new_wav.timestamp = self.timestamp[start:end]
        new_wav.wav = self.wav[start:end, :]
        new_wav.fiber_id = self.fiber_id
        if self.wls is not None:
            new_wav.wls = self.wls.iloc[start:end,:]
        return new_wav

    def to_wls(self):
        wav_df = pd.DataFrame(index=self.timestamp, data=self.wav)
        wls = detrend_df(wav_df) * 1000
        wls = remove_outliers_df(wls, outlier_threshold=500)
        wls.sort_index(inplace=True)
        # wls = detrend_array(self.wav) * 1000
        self.wls = wls
        return wls

    def pick_sensors(self, sensors=[0,1,2,3,4,5,6,7,8,9,10]):
        # TODO
        self.wav = None
        self.wls = None
        pass


    def plot_agg_history(self, pick_range=[0, 10000]):
        if pick_range is None:
            start = 0
            end = len(self.timestamp)
        else:
            start, end = pick_range
            end = min(len(self.timestamp), end)

        if self.wls is not None:
            wls = self.wls.iloc[start:end, :]
        else:
            time_index = self.timestamp[start:end]
            # wls = detrend_array(self.wav[start:end,:])*1000
            wav_df = pd.DataFrame(index=time_index, data=self.wav[start:end, :])
            wls = detrend_df(wav_df) * 1000

        agg_df = wls.min(axis=1)
        # df = pd.DataFrame(index=time_index, data=agg_data)
        logger.info('plotting wls...')
        fig = px.line(agg_df)
        plot(fig, filename='agg_history.html')
        return fig
        # plt.show()

    def plot_agg_profile(self, pick_range=[0.5000]):
        if pick_range is None:
            start = 0
            end = len(self.timestamp)
        else:
            start, end = pick_range
            end = min(len(self.timestamp), end)

        if self.wls is not None:
            wls = self.wls.iloc[start:end, :]
        else:
            time_index = self.timestamp[start:end]
            # wls = detrend_array(self.wav[start:end,:])*1000
            wav_df = pd.DataFrame(index=time_index, data=self.wav[start:end, :])
            wls = detrend_df(wav_df) * 1000
            wls.columns = [f'S{i + 1}' for i in range(wav_df.shape[1])]

        agg_df = wls.min(axis=0)
        # df = pd.DataFrame(index=time_index, data=agg_data)
        logger.info('plotting wls...')
        fig = px.line(agg_df)
        plot(fig, filename='agg_profile.html')
        return fig


if __name__ == "__main__":
    # from pydea.datamodels.datamodel import Wav
    data_dir = Path(
        r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\HighFrequency333Hz_20201213\HighFreq_333Hz\wav')
    wav_file = data_dir / 'wav_20201212_195900_F03_UTC.npz'
    # data = np.load(wav_file, allow_pickle=True)
    wav = Wav(wav_file)
    wav.to_wls()
    wav.plot_agg_history(pick_range=None)

    # wav.plot_agg_profile(pick_range=None)

    # wav = Wav(wav_file)
