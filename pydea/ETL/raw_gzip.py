from datetime import timedelta, datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import gzip


channel_starts = [8, 64, 120, 176]
FBGS_TIME_SHIFT_TO_UTC = timedelta(hours=1)
TIME_SHIFT = timedelta(hours=14,minutes=53,seconds=24)

#%% M80 time shift
# FBGS time: 2020/11/24 17:35:13 = PC time: 11/25 08:29:25 = Internet Melbourne time 11/25 08:29:22
M80_TIME_SHIFT_TO_UTC = timedelta(hours=3,minutes=53,seconds=9) #M80
Francis_TIME_SHIFT_TO_UTC = timedelta(hours=0,minutes=57,seconds=52) #Francis st.

def interpolate_time_index(time_index, line_number, time_shift= TIME_SHIFT):
    t_start = pd.to_datetime(time_index[0]).to_pydatetime()
    t_end = pd.to_datetime(time_index[-1]).to_pydatetime()
    print(f'start:{t_start}, end {t_end}')
    delta_line_number = line_number[-1] - line_number[0]
    # ln_df = pd.DataFrame(line_number.values)
    # ln_diff = ln_df.diff()
    # print(ln_diff)
    # if (ln_diff<0).any():
    #     print('line number is not monotanic')
    line_number0 = line_number - line_number[0]
    dt = (t_end - t_start).total_seconds() / delta_line_number
    # t_start_melbourne = t_start + time_shift
    t_start_shifted = t_start + time_shift
    t_start_utc = t_start_shifted.replace(tzinfo=timezone.utc)
    new_time_index = [t_start_utc  + timedelta(seconds=dt * line_number) for line_number in line_number0]
    return new_time_index

def parse_raw_data(raw_df, channel_starts=channel_starts, interpolate_timestamp=True, time_shift=FBGS_TIME_SHIFT_TO_UTC):
    dataset = {}
    line_number = raw_df['col2']
    time_index = pd.to_datetime(raw_df.index, errors='ignore')
    # time_index = shift_datetime_into_utc(line_number, time_index)
    raw_df.index = time_index
    if interpolate_timestamp:
        raw_df.index = interpolate_time_index(time_index, line_number, time_shift=time_shift)
    # line_number.to_pickle('francis_raw_line_number.pickle')

    for i, ch_start in enumerate(channel_starts):
        channel_id = i + 1
        ch_df = raw_df.iloc[:, ch_start:ch_start + 25]
        ch_df.columns = [f'sensor{i + 1}' for i in range(25)]
        dataset[f'ch_{channel_id}'] = ch_df
    return dataset, line_number

def read_raw_data(data_file, timezone='UTC'):
    # first_row = pd.read_csv(data_file, delimiter='\t', error_bad_lines=False, nrows=1, low_memory=False)
    # len_data = first_row.shape[1]
    # # col_names = [f'col{i}' for i in range(len_data)]
    if str(data_file).endswith('.gzip'):
        data_file = gzip.open(data_file)

    df = pd.read_table(data_file, delimiter='\t', error_bad_lines=False, prefix='col', header=None,
                       parse_dates={'timestamp': [0, 1]},
                       keep_date_col=False, low_memory=False, engine='c'
                       )  # names=col_names,infer_datetime_format=True,cache_dates=True
    time_index = pd.to_datetime(df.index, errors='coerce')
    df.index = time_index
    df.set_index('timestamp', inplace=True)
    return df

def read_files(data_files):
    dfs = []
    for data_file in data_files:
        # print(f'save to npz: {data_file}')
        try:
            df = read_raw_data(data_file)
            dfs.append(df)
        except:
            print('unable to read raw data. ignore')

    combined_df = pd.concat(dfs)
    return combined_df

def remove_bad_rows(wav):
    new_wav = []
    valid_index = []
    for i, row in enumerate(wav):
        try:
            row = row.astype('float')
        except:
            continue
        new_wav.append(row)
        valid_index.append(i)
    new_wav = np.array(new_wav)
    new_wav.astype('float')
    return new_wav, valid_index

def extract_raw_files_into_wav_format(batch_files,
                                      channel_starts,
                                      time_shift_to_utc=FBGS_TIME_SHIFT_TO_UTC,
                                      time_range=None,
                                      out_dir=None,
                                      save_to_pickle=False):
    df = read_files(batch_files)
    data_dir = batch_files[0].parent
    if out_dir is None:
        out_dir = data_dir/'wav'
        out_dir.mkdir(parents=True, exist_ok=True)

    dataset, line_number = parse_raw_data(df, channel_starts=channel_starts,
                                          interpolate_timestamp=True,
                                          time_shift=time_shift_to_utc,
                                          )
    t_start = pd.to_datetime(df.index[0])
    tstart_str = datetime.strftime(t_start,'%Y%m%d_%H%M%S')

    for ch, ch_df in dataset.items():
        ch_num = ch[3:]
        fiber_id = f'{int(ch_num):02d}'
        wav_file_name = f'wav_{tstart_str}_F{fiber_id}_UTC.npz'
        # ch_file_name = f'{filename}_{ch}.pickle'
        wav = ch_df.values
        ln = line_number.values
        ts = ch_df.index.values
        wav1, valid_ind = remove_bad_rows(wav)
        ts1 = ts[valid_ind]
        ln1 = ln[valid_ind]

        if time_range is not None:
            t0, t1 = time_range
            wav1 = wav1[t0:t1]

        np.savez_compressed(out_dir/wav_file_name,
                            timestamp=ts1,
                            wav=wav1,
                            fiber_id=fiber_id,
                            )

        if save_to_pickle:
            df1 = pd.DataFrame(index=ts1, data=wav1)
            df1.columns = [f'sensor{i+1}' for i in range(len(df1.columns))]
            df1.to_pickle(data_dir / f'wav_{tstart_str}_F{fiber_id}_UTC.pickle')
        #
        # # wls = 1000*detrend_df(df1)
        # # wls = remove_outlier(wls,outlier_threshold=1000)
        # wav = df1
        # if time_range is not None:
        #     t0, t1 = time_range
        #     wav = wav[t0:t1]
        #
        # wav.to_pickle(data_dir/ch_file_name)

if __name__ == '__main__':
    #%% Francis st
    TIME_SHIFT_TO_UTC = Francis_TIME_SHIFT_TO_UTC
    data_dir = Path(
        r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\Francis\VIPER VIM validation\driveeasy_data\raw_gzip')
    data_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\Francis\VIPER VIM validation\driveeasy_data\raw_gzip\melbourne_time_20201120_0900AM')
    data_files = list(data_dir.glob('*.gzip'))
    extract_raw_files_into_wav_format(data_files,
                                      channel_starts=channel_starts,
                                      time_shift_to_utc=TIME_SHIFT_TO_UTC,
                                      time_range=None,
                                      out_dir=None,
                                      save_to_pickle=False)


    #%% M80
    # TIME_SHIFT_TO_UTC = M80_TIME_SHIFT_TO_UTC
    # data_dir = Path(r'C:\Users\hyu\PARC\Fibridge-PARC - General\Drive Easy\AustraliaDeploy\M80\TIRTLE validation\20201124\driveeasy_raw\melbourne_time_0700AM\melbourne_time_20201125_0700AM')
    # data_files = list(data_dir.glob('*.gzip'))
    # extract_raw_files_into_wav_format(data_files,
    #                                   channel_starts=channel_starts,
    #                                   time_shift_to_utc=TIME_SHIFT_TO_UTC,
    #                                   time_range=None,
    #                                   out_dir=None,
    #                                   save_to_pickle=False)

    # df = read_files(data_files)
    #
    # print(df.head())
#%% load data
# data_file = 'wav_20201119_140113_F01_UTC.npz'
# d = np.load(data_dir/data_file)