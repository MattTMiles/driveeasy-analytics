from collections import namedtuple
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pprint import pprint
import gzip
import numpy as np
from collections import defaultdict
import pandas as pd
from loguru import logger
import pytz
from tqdm import tqdm
from pydea.preprocessing.shift_time import interpolate_time_index_from_line_number

from pydea.defaults import M80_TIME_SHIFT_TO_UTC, Francis_TIME_SHIFT_TO_UTC
# M80_TIME_SHIFT_TO_UTC = timedelta(hours=3,minutes=53,seconds=9) #M80
# Francis_TIME_SHIFT_TO_UTC = timedelta(hours=0,minutes=57,seconds=52) #Francis st.

def convert_raw_gzip_into_text_file(raw_file, out_file=None):
    raw_file = Path(raw_file)
    with gzip.GzipFile(raw_file, 'r') as f:
        data = f.read()
        if out_file is None:
            file_stem = raw_file.stem
            out_file = raw_file.parent / f'{file_stem}.txt'
        print(f'txt file saved to: {out_file}')
        with open(out_file, 'wb') as out_f:
            out_f.write(data)

class FbgsDataRow:
    """
    Data structure for Fbgs data.

    Attributes:
        date: timestamp date
        time timestamp time (HH:MM:SS)
        line_number: Int number
        num_channels: number of active channels used on the FBGS device.
        channel_data: a Dict, where keys are channel names (eg. 'channel_1'), values are list of wavelengths at one timestamp.

    """

    def __init__(self, date, time, line_number, channel_list, channel_data, error_code=[0, 0, 0, 0]):
        self.date = date
        self.time = time
        self.line_number = line_number
        self.num_channels = len(channel_list)
        self.channel_data = channel_data
        self.channel_list = channel_list
        # self.strain = []
        self.error_code = error_code
        # self.active_channels = [1, 2, 3, 4]

    def to_dataframe_dict(self):
        for ch in self.channel_list:
            ch_data = self.channel_data[f'channel_{ch}']
            df = pd.DataFrame()

    def validate_sensors(self):
        pass

    def to_dataframe(self):
        pass

class FbgsDataBlock:
    def __init__(self, filename=None, to_wav_data=False, interpolate_time_index=True, time_shift_to_UTC=None):
        self.data_block = []
        self.wav_dataset = defaultdict(dict)
        self.interpolate_time_index = interpolate_time_index
        self.time_shift_to_UTC = time_shift_to_UTC
        self.bad_data_count = 0
        if filename is not None:
            self.from_gzip_file(gzip_file=filename, to_wav_data=to_wav_data,parser=FbgsParser())

    def from_gzip_file(self, gzip_file, to_wav_data=False, parser=FbgsParser()):
        # self.data_block = []
        logger.info(f'reading file {gzip_file}')
        with gzip.GzipFile(gzip_file, 'r') as f:
            # for i in range(5):
            for line in f:
                # line = next(f)
                # for line in f[0:3]:
                parsed_raw = parser.parse(line)
                if parsed_raw is None:
                    self.bad_data_count += 1
                else:
                    self.data_block.append(parsed_raw)
        if to_wav_data:
            self.to_wav_data(interpolate_time_index=self.interpolate_time_index, time_shift_to_UTC=self.time_shift_to_UTC)
        logger.info(f'number of bad data: {self.bad_data_count}')

    def __repr__(self):
        # print('print fbgs data')
        return f'Number of data points: {len(self.data_block)}. \n Number of bad data points: {self.bad_data_count}'

    def from_datarow(self, fbgs_row):
        self.data_block = [fbgs_row]

    def append_block(self, fbgs_block):
        if isinstance(fbgs_block, FbgsDataBlock):
            self.data_block.extend(fbgs_block.data_block)
            logger.info('FBGS data block attached.')
        else:
            logger.warning('fbgs_block is not an instance of FbgsDataBlock type.')

    def append_row(self, fbgs_row):
        self.data_block.append(fbgs_row)

    def to_dataframe(self):
        pass

    def save_wav_file(self, out_dir = None):
        if len(self.wav_dataset) == 0:
            logger.warning('wav data empty. Cannot save to file.')
            return

        if not out_dir:
            out_dir = Path('../ETL')

        for ch_id, ch_dataset in self.wav_dataset.items():
            ch_timestamp = ch_dataset['timestamp']
            ch_line_number = ch_dataset['line_number']
            ts0 = ch_timestamp[0]
            ts0_str = ts0.strftime('%Y%m%d_%H%M%S')
            ch_wav = np.array(ch_dataset['wav'])
            ch_fiber_id = ch_id[8:]  # channel_1, channel_2
            # if not filename:
            filename = f'wav_{ts0_str}_F{int(ch_fiber_id):02d}'
            np.savez_compressed(out_dir / filename,
                                timestamp=ch_timestamp,
                                line_number=ch_line_number,
                                wav=ch_wav,
                                fiber_id=ch_fiber_id)
            logger.info(f'{ch_id} data saved to {out_dir}/{filename}')

    def to_wav_data(self,interpolate_time_index,time_shift_to_UTC):
        """
        wav_dataset['channel_3']: {'timestamp':timestamp,'wav':wav}
        Returns:

        """
        logger.info('start convert raw bytes into wav data.')
        if len(self.data_block) == 0:
            logger.warning('data block empty; cannot convert to wav data.')
            return self.wav_dataset

        wav_dataset = defaultdict(dict)
        for fbgs_row in tqdm(self.data_block):
            date = fbgs_row.date
            time = fbgs_row.time
            row_timestamp = pd.Timestamp(f'{date} {time}')
            row_line_number = fbgs_row.line_number

            channel_data = fbgs_row.channel_data
            for channel, ch_data in channel_data.items():
                if ch_data.error_code == ['0', '0', '0', '0']:  # ignore data with errors:
                    if channel not in wav_dataset:
                        wav_dataset[channel] = {'timestamp':np.array([]),'wav':[],'line_number':np.array([])}
                    # ch_dataset = wav_dataset['channel']
                # wav_dataset[channel]['wav'] = np.vstack((wav_dataset[channel]['wav'],ch_data.wavelength))

                    wav_dataset[channel]['wav'].append(ch_data.wavelength)
                    # wav_dataset[channel]['wav'] = np.append(wav_dataset[channel]['wav'],ch_data.wavelength)
                    wav_dataset[channel]['timestamp'] = np.append(wav_dataset[channel]['timestamp'],row_timestamp)
                    wav_dataset[channel]['line_number'] = np.append(wav_dataset[channel]['line_number'],row_line_number)
                # wav_dataset[channel]['wav'].append(ch_data.wavelength)
                # wav_dataset[channel]['timestamp'].append(row_timestamp)
                # wav_dataset[channel]['line_number'].append(row_line_number)

        if interpolate_time_index:
            logger.info('start interpolating time index from line number.')
            for ch_id, ch_dataset in tqdm(wav_dataset.items()):
                ch_timestamp = ch_dataset['timestamp']
                ch_line_number = ch_dataset['line_number']
                new_timestamp = interpolate_time_index_from_line_number(time_index=ch_timestamp,
                                                       line_number=ch_line_number,
                                                       time_shift=time_shift_to_UTC,
                                                       out_time_zone = timezone.utc)
                logger.info(f'old time index: {ch_timestamp[0]}, new time index: {new_timestamp[0]}')
                ch_dataset['timestamp'] = new_timestamp
            logger.info('finished interpolating time index from line number.')
        self.wav_dataset = wav_dataset
        return wav_dataset

    def to_wls_data(self):
        pass

    def validate_sensors(self):
        pass

if __name__ == '__main__':
    data_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\HighFrequency333Hz_20201213\HighFreq_333Hz')
    # file1 = data_dir/'raw_2020-12-12_20-00-41.gzip'
    files = sorted(list(data_dir.glob('*.gzip')))
    print(f'first fiel: {files[0]}')
    print(f'last file: {files[-1]}')

    fbgs_raw = FbgsDataBlock()
    for file in tqdm(files):
        new_fbgs_block = FbgsDataBlock(file,to_wav_data=False, interpolate_time_index=False, time_shift_to_UTC=M80_TIME_SHIFT_TO_UTC)
        print(new_fbgs_block.data_block[0].line_number)
        print(new_fbgs_block.data_block[-1].line_number)
        fbgs_raw.append_block(new_fbgs_block)
    fbgs_raw.to_wav_data(interpolate_time_index=True,time_shift_to_UTC=M80_TIME_SHIFT_TO_UTC)
    # fbgs_raw = FbgsDataBlock(file1,interpolate_time_index=True, time_shift_to_UTC=M80_TIME_SHIFT_TO_UTC)
    print(fbgs_raw)

    fbgs_raw.save_wav_file()
    # parser = FbgsParser()
    #
    # fbgs_data = FbgsDataBlock()
    # bad_count = 0
    # with gzip.GzipFile(file1,'r') as f:
    #     # for i in range(5):
    #     for line in f:
    #         # line = next(f)
    #     # for line in f[0:3]:
    #         parsed_raw = parser.parse(line)
    #         if parsed_raw is None:
    #             bad_count += 1
    #         else:
    #             fbgs_data.append_row(parsed_raw)
    # wav_dataset = fbgs_data.to_wav_data()
    # print(f'number of bad data: {bad_count}')
    # print(wav_dataset.keys())
    #
    #%% test load data
    d3 = np.load(Path(r'/pydea/ETL/wav_20201212_195900_F03.npz'), allow_pickle=True)
    #         # print(parsed_raw)