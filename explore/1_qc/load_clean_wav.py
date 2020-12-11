import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import datetime


def read_npz_file(filename):
    data = np.load(filename, allow_pickle=True)
    df = pd.DataFrame(data=data['wav'], index=data['timestamp'])
    df.columns = [f'sensor{i+1}' for i in range(25)]
    return df


def clean_wav(fiber):
    df_fiber = fiber
    check = df_fiber.index.to_series().diff().dt.total_seconds()
    df = pd.DataFrame(index=(df_fiber[check.values>0.006].index-datetime.timedelta(seconds=0.005)), columns=df_fiber[check.values>0.006].columns)
    df_fiber = df_fiber.append(df, ignore_index=False)
    df_fiber = df_fiber.sort_index()
    df_fiber.reset_index(inplace=True)
    df_fiber.columns = ['timestamp'] + [f'sensor{i+1}' for i in range(25)]
    return df_fiber


def find_outliers(df, percent_low=0.001, percent_high=0.99999):
    # remove outliers; There should be better way to do it.
    y = df['sensor1']
    removed_outliers = (y.between(y.quantile(.000001), y.quantile(.9999999)))
    #     print(removed_outliers.sum())
    for i in range(df.shape[1] - 1):
        y = df[f'sensor{i + 1}']
        if np.max([y.max() - y.mean(), y.mean() - y.min()]) > 50 * y.std():
            removed_outliers = y.between(y.quantile(percent_low), y.quantile(percent_high)) & removed_outliers
    index_names = df[~removed_outliers].index
    return index_names


def remove_outliers_from_paired_fibers(outlier_ch1, outlier_ch2, df_1, df_2):
    total_outliers = list(dict.fromkeys(outlier_ch1.to_list() + outlier_ch2.to_list()))
    df_1.loc[total_outliers, [f'sensor{i + 1}' for i in range(25)]] = np.nan
    df_2.loc[total_outliers, [f'sensor{i + 1}' for i in range(25)]] = np.nan

    df_1.interpolate(inplace=True)
    df_2.interpolate(inplace=True)
    return df_1, df_2


if __name__ == '__main__':

    filename_1 = r'C:\Users\qchen\PARC\Fibridge-PARC - Drive Easy\AustraliaDeploy\Calibration Test 20201201\M80\M80_DriveEasy\wav\wav_20201202_024632_F03_UTC.npz'
    filename_2 = r'C:\Users\qchen\PARC\Fibridge-PARC - Drive Easy\AustraliaDeploy\Calibration Test 20201201\M80\M80_DriveEasy\wav\wav_20201202_024632_F04_UTC.npz'

    df_1 = read_npz_file(filename_1)
    df_2 = read_npz_file(filename_2)

    df_1 = clean_wav(df_1)
    df_2 = clean_wav(df_2)

    outliers_1 = find_outliers(df_1, percent_low=0.004, percent_high=0.99999)
    outliers_2 = find_outliers(df_2, percent_low=0.004, percent_high=0.99999)

    df_1, df_2 = remove_outliers_from_paired_fibers(outliers_1, outliers_2, df_1, df_2)

    # fix missing data in the head
    df_1 = df_1.fillna(method='bfill')
    df_2 = df_2.fillna(method='bfill')

    print('Head of df_1:')
    print(df_1.head(10))

    print('Head of df_2:')
    print(df_2.head(10))

    # df_1.plot(y=['sensor20', 'sensor21', 'sensor22', 'sensor23', 'sensor24', 'sensor25'])
