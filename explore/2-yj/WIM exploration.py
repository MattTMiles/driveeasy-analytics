import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import scipy
import scaleogram as scg 
import seaborn as sns
import sklearn

font = {'family' : 'times new roman',
        'weight' : 'normal',
        'size'   : 12}

plt.rc('font', **font)

filepath = r'C:\Users\Jin Yan\OneDrive - PARC\FiBridge\Phase_3\DriveEasy\Calibration Test 20201201\M80\Weigh_in_motion\test_data'
filename = filepath+'/lane2_run1_rigid.pkl'
fibers = pd.read_pickle(filename)

fiber_signals = {'lane2run1rigid': pd.read_pickle(filepath+'/lane2_run1_rigid.pkl'),
                 'lane2run1semi': pd.read_pickle(filepath+'/lane2_run1_semi.pkl'),
                 'lane2run1van': pd.read_pickle(filepath+'/lane2_run1_van.pkl'),
                 'lane2run2rigid': pd.read_pickle(filepath+'/lane2_run2_rigid.pkl'),
                 'lane2run2semi': pd.read_pickle(filepath+'/lane2_run2_semi.pkl'),
                 'lane2run2van': pd.read_pickle(filepath+'/lane2_run2_van.pkl'),
                 'lane2run3rigid': pd.read_pickle(filepath+'/lane2_run3_rigid.pkl'),
                 'lane2run3semi': pd.read_pickle(filepath+'/lane2_run3_semi.pkl'),
                 'lane2run3van': pd.read_pickle(filepath+'/lane2_run3_van.pkl'),
                 'lane5run3rigid': pd.read_pickle(filepath+'/lane5_run3_rigid.pkl'),
                 'lane5run3semi': pd.read_pickle(filepath+'/lane5_run3_semi.pkl'),
                 'lane5run3van': pd.read_pickle(filepath+'/lane5_run3_van.pkl')}


#%%
lane5run1rigid = fiber_signals.get('lane5run1rigid')
lane5run2rigid = fiber_signals.get('lane5run2rigid')
lane5run3rigid = fiber_signals.get('lane5run3rigid')

lane2run1rigid = fiber_signals.get('lane2run1rigid')
lane2run2rigid = fiber_signals.get('lane2run2rigid')
lane2run3rigid = fiber_signals.get('lane2run3rigid')

lane2run1semi = fiber_signals.get('lane2run1semi')
lane2run2semi = fiber_signals.get('lane2run2semi')
lane2run3semi = fiber_signals.get('lane2run3semi')

lane2run1van = fiber_signals.get('lane2run1van')
lane2run2van = fiber_signals.get('lane2run2van')
lane2run3van = fiber_signals.get('lane2run3van')

lane5run3rigid = fiber_signals.get('lane5run3rigid')
lane5run3semi = fiber_signals.get('lane5run3semi')
lane5run3van = fiber_signals.get('lane5run3van')

#%% Plot the 4 peak sensing points and time-frequency plot using CWT
# choose default wavelet function, currently find out 'gaus7' is the best
scg.set_default_wavelet('gaus7')

ncol = 3
nrow = 3

# Rigid
figsize = (15, 1.5*6)
fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
plt.subplots_adjust(hspace=0.5, wspace=0.5 )
axes = [ item for sublist in axes for item in sublist ] # flatten list

for i, df in enumerate([lane2run1rigid, lane2run2rigid, lane2run3rigid]):
    ax=axes[i]
    signal_raw_trailing_left = df.iloc[:,-9].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_trailing_right = df.iloc[:,-3].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_left = df.iloc[:,31].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_right = df.iloc[:,37].interpolate(method='linear', axis=0).ffill().bfill()
    combined_2peaks_trailing = (signal_raw_trailing_left+signal_raw_trailing_right)/2
    combined_2peaks_leading = (signal_raw_leading_left+signal_raw_leading_right)/2
    signal_length = len(signal_raw_trailing_left)
    # range of scales to perform the transform
    scales = scg.periods2scales( np.arange(1, 5,0.01) )
    x_values_wvt_arr = range(0,signal_length,1)
    ax.plot(x_values_wvt_arr, signal_raw_trailing_left, linewidth=2, label = 'left trailing')
    ax.plot(x_values_wvt_arr, signal_raw_trailing_right, linewidth=2, label = 'right trailing')
    ax.plot(x_values_wvt_arr, signal_raw_leading_left, linewidth=2, label = 'left leading')
    ax.plot(x_values_wvt_arr, signal_raw_leading_right, linewidth=2, label = 'right leading')
    leg = ax.legend()
    ax.set_ylim(-0.05,0.03)
    
    ax=axes[i+3]
    ax.plot(x_values_wvt_arr, combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(x_values_wvt_arr, combined_2peaks_leading, linewidth=2, label = 'leading')
    leg = ax.legend()
    ax.set_ylim(-0.05,0.03)
    
    ax=axes[i+6]
    scg.cws(combined_2peaks_leading[:signal_length], scales=scales, coi = False, ylabel="Period", xlabel="Time",
        title='Run '+str(i+1)+' rigid scaleogram', ax=ax)
    
# Semi
figsize = (15, 1.5*6)
fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
plt.subplots_adjust(hspace=0.5, wspace=0.5 )
axes = [ item for sublist in axes for item in sublist ] # flatten list

for i, df in enumerate([lane2run1semi, lane2run2semi, lane2run3semi]):
    ax=axes[i]
    signal_raw_trailing_left = df.iloc[:,-9].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_trailing_right = df.iloc[:,-3].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_left = df.iloc[:,31].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_right = df.iloc[:,37].interpolate(method='linear', axis=0).ffill().bfill()
    combined_2peaks_trailing = (signal_raw_trailing_left+signal_raw_trailing_right)/2
    combined_2peaks_leading = (signal_raw_leading_left+signal_raw_leading_right)/2
    signal_length = len(signal_raw_trailing_left)
    # range of scales to perform the transform
    scales = scg.periods2scales( np.arange(1, 5,0.01) )
    x_values_wvt_arr = range(0,signal_length,1)
    ax.plot(x_values_wvt_arr, signal_raw_trailing_left, linewidth=2, label = 'left trailing')
    ax.plot(x_values_wvt_arr, signal_raw_trailing_right, linewidth=2, label = 'right trailing')
    ax.plot(x_values_wvt_arr, signal_raw_leading_left, linewidth=2, label = 'left leading')
    ax.plot(x_values_wvt_arr, signal_raw_leading_right, linewidth=2, label = 'right leading')
    leg = ax.legend()
    ax.set_ylim(-0.05,0.03)
    
    ax=axes[i+3]
    ax.plot(x_values_wvt_arr, combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(x_values_wvt_arr, combined_2peaks_leading, linewidth=2, label = 'leading')
    leg = ax.legend()
    ax.set_ylim(-0.05,0.03)
    
    ax=axes[i+6]
    scg.cws(combined_2peaks_leading[:signal_length], scales=scales, coi = False, ylabel="Period", xlabel="Time",
        title='Run '+str(i+1)+' rigid scaleogram', ax=ax)
    
# Van
figsize = (15, 1.5*6)
fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
plt.subplots_adjust(hspace=0.5, wspace=0.5 )
axes = [ item for sublist in axes for item in sublist ] # flatten list

for i, df in enumerate([lane2run1van, lane2run2van, lane2run3van]):
    ax=axes[i]
    signal_raw_trailing_left = df.iloc[:,-8].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_trailing_right = df.iloc[:,-3].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_left = df.iloc[:,32].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_right = df.iloc[:,37].interpolate(method='linear', axis=0).ffill().bfill()
    combined_2peaks_trailing = (signal_raw_trailing_left+signal_raw_trailing_right)/2
    combined_2peaks_leading = (signal_raw_leading_left+signal_raw_leading_right)/2
    signal_length = len(signal_raw_trailing_left)
    # range of scales to perform the transform
    scales = scg.periods2scales( np.arange(1, 5,0.01) )
    x_values_wvt_arr = range(0,signal_length,1)
    ax.plot(x_values_wvt_arr, signal_raw_trailing_left, linewidth=2, label = 'left trailing')
    ax.plot(x_values_wvt_arr, signal_raw_trailing_right, linewidth=2, label = 'right trailing')
    ax.plot(x_values_wvt_arr, signal_raw_leading_left, linewidth=2, label = 'left leading')
    ax.plot(x_values_wvt_arr, signal_raw_leading_right, linewidth=2, label = 'right leading')
    leg = ax.legend()
    ax.set_ylim(-0.05,0.03)
    
    ax=axes[i+3]
    ax.plot(x_values_wvt_arr, combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(x_values_wvt_arr, combined_2peaks_leading, linewidth=2, label = 'leading')
    leg = ax.legend()
    ax.set_ylim(-0.05,0.03)
    
    ax=axes[i+6]
    scg.cws(combined_2peaks_leading[:signal_length], scales=scales, coi = False, ylabel="Period", xlabel="Time",
        title='Run '+str(i+1)+' rigid scaleogram', ax=ax)
    
    
    
#%% Features

# prepare the feature table
df_features = pd.DataFrame(columns=['steer_leading_prominence','steer_trailing_prominence',
                                                                'steer_leading_height','steer_trailing_height',
                                                                'steer_leading_width','steer_trailing_width',
                                                                'steer_leading_area','steer_trailing_area',
                                                                'drive_leading_prominence','drive_trailing_prominence',
                                                                'drive_leading_height','drive_trailing_height',
                                                                'drive_leading_width','drive_trailing_width',
                                                                'drive_leading_area','drive_trailing_area',
                                                                'trailer_leading_prominence','trailer_trailing_prominence',
                                                                'trailer_leading_height','trailer_trailing_height',
                                                                'trailer_leading_width','trailer_trailing_width',
                                                                'trailer_leading_area','trailer_trailing_area',
                                                                'steer_weight','drive_weight','trailer_weight',
                                                                'speed','type'])
                           # index=['lane2run1rigid','lane2run2rigid','lane2run3rigid',
                           #        'lane2run1semi','lane2run2semi','lane2run3semi',
                           #        'lane2run1van','lane2run2van','lane2run3van'])
df_features = df_features.fillna(0) # with 0s rather than NaNs

# define the filter parameteres
fc = 10  # Cut-off frequency of the filter
w = fc / (200 / 2) # Normalize the frequency
b, a = scipy.signal.butter(5, w, 'low')


ncol = 3
nrow = 2

# Rigid
figsize = (15, 1.5*6)
fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
axes = [ item for sublist in axes for item in sublist ] # flatten list

lane2list = [lane2run1rigid, lane2run2rigid, lane2run3rigid]
for i, df in enumerate(lane2list):
    ax=axes[i]
    signal_raw_trailing_left = df.iloc[:,-9].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_trailing_right = df.iloc[:,-3].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_left = df.iloc[:,31].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_right = df.iloc[:,37].interpolate(method='linear', axis=0).ffill().bfill()
    combined_2peaks_trailing = (signal_raw_trailing_left+signal_raw_trailing_right)/2
    combined_2peaks_leading = (signal_raw_leading_left+signal_raw_leading_right)/2
    signal_length = len(signal_raw_trailing_left)
    # range of scales to perform the transform
    scales = scg.periods2scales( np.arange(1, 5,0.01) )
    x_values_wvt_arr = range(0,signal_length,1)
    ax.plot(x_values_wvt_arr, combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(x_values_wvt_arr, combined_2peaks_leading, linewidth=2, label = 'leading')
    ax.set_xlabel('sample')
    ax.set_ylabel('wavelength shift (nm)')
    ax.title.set_text('raw combined L+R')
    leg = ax.legend()
    
    ax=axes[i+3]
    combined_2peaks_trailing = signal.filtfilt(b, a, combined_2peaks_trailing)
    combined_2peaks_leading = signal.filtfilt(b, a, combined_2peaks_leading)
    time_peaks_s1, peak_prop_s1 = signal.find_peaks(-combined_2peaks_trailing, distance=3, prominence =0.001, width=0.00005, height=0.00001)
    time_peaks_s2, peak_prop_s2 = signal.find_peaks(-combined_2peaks_leading, distance=3, prominence =0.001, width=0.00005, height=0.00001)
    ax.plot(x_values_wvt_arr, -combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(time_peaks_s1, -combined_2peaks_trailing[time_peaks_s1], "x")    
    ax.vlines(x=time_peaks_s1, ymin=-combined_2peaks_trailing[time_peaks_s1] - peak_prop_s1["prominences"],
        ymax = -combined_2peaks_trailing[time_peaks_s1])    
    ax.hlines(y=peak_prop_s1["width_heights"], xmin=peak_prop_s1["left_ips"],
                xmax=peak_prop_s1["right_ips"])
    
    ax.plot(x_values_wvt_arr, -combined_2peaks_leading, linewidth=2, label = 'leading')
    ax.plot(time_peaks_s2, -combined_2peaks_leading[time_peaks_s2], ".")
    ax.set_xlabel('sample')
    ax.set_ylabel('wavelength shift (nm)')
    ax.title.set_text('filtered combined L+R')
    ax.vlines(x=time_peaks_s2, ymin=-combined_2peaks_leading[time_peaks_s2] - peak_prop_s2["prominences"],
        ymax = -combined_2peaks_leading[time_peaks_s2])
    ax.hlines(y=peak_prop_s2["width_heights"], xmin=peak_prop_s2["left_ips"],
                xmax=peak_prop_s2["right_ips"])
    leg = ax.legend()
    
    tempDf = pd.DataFrame(data = [peak_prop_s1["peak_heights"]], 
                          columns=['steer_trailing_height','drive_trailing_height'])
    tempDf[['steer_trailing_width','drive_trailing_width']] = peak_prop_s1["widths"]
    tempDf[['steer_trailing_prominence','drive_trailing_prominence']] = peak_prop_s1["prominences"]
    tempDf[['steer_leading_height','drive_leading_height']] = peak_prop_s2["peak_heights"]
    tempDf[['steer_leading_width','drive_leading_width']] = peak_prop_s2["widths"]
    tempDf[['steer_leading_prominence','drive_leading_prominence']] = peak_prop_s2["prominences"]
    tempDf[['steer_weight','drive_weight','trailer_weight']]=np.array([4.82,8.26,0])
    bases = np.unique(np.concatenate((peak_prop_s2['right_bases'],peak_prop_s2['left_bases'])))
    tempDf['steer_leading_area'] = -np.trapz(combined_2peaks_leading[bases[0]:bases[1]])
    tempDf['drive_leading_area'] = -np.trapz(combined_2peaks_leading[bases[1]:bases[2]])
    bases = np.unique(np.concatenate((peak_prop_s1['right_bases'],peak_prop_s1['left_bases'])))
    tempDf['steer_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[0]:bases[1]])
    tempDf['drive_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[1]:bases[2]])
    # tempDf[['trailer_leading_area','trailer_trailing_area']] = np.array([0,0])
    speed = 2.5/(signal.correlate(combined_2peaks_leading, combined_2peaks_trailing).argmax()-signal_length)*200*3.6
    tempDf[['speed','type']]=np.array([speed,1])
    df_features=pd.concat([df_features,tempDf])

    
# Semi
figsize = (15, 1.5*6)
fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
axes = [ item for sublist in axes for item in sublist ] # flatten list

for i, df in enumerate([lane2run1semi, lane2run2semi, lane2run3semi]):
    ax=axes[i]
    signal_raw_trailing_left = df.iloc[:,-9].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_trailing_right = df.iloc[:,-3].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_left = df.iloc[:,31].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_right = df.iloc[:,37].interpolate(method='linear', axis=0).ffill().bfill()
    combined_2peaks_trailing = (signal_raw_trailing_left+signal_raw_trailing_right)/2
    combined_2peaks_leading = (signal_raw_leading_left+signal_raw_leading_right)/2
    signal_length = len(signal_raw_trailing_left)
    # range of scales to perform the transform
    scales = scg.periods2scales( np.arange(1, 5,0.01) )
    x_values_wvt_arr = range(0,signal_length,1)
    ax.plot(x_values_wvt_arr, combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(x_values_wvt_arr, combined_2peaks_leading, linewidth=2, label = 'leading')
    ax.set_xlabel('sample')
    ax.set_ylabel('wavelength shift (nm)')
    ax.title.set_text('raw combined L+R')
    leg = ax.legend()
    
    ax=axes[i+3]
    combined_2peaks_trailing = signal.filtfilt(b, a, combined_2peaks_trailing)
    combined_2peaks_leading = signal.filtfilt(b, a, combined_2peaks_leading)
    time_peaks_s1, peak_prop_s1 = signal.find_peaks(-combined_2peaks_trailing, distance=3, prominence =0.0006, width=0.00005, height=0.00001)
    time_peaks_s2, peak_prop_s2 = signal.find_peaks(-combined_2peaks_leading, distance=3, prominence =0.0006, width=0.00005, height=0.00001)
    ax.plot(x_values_wvt_arr, -combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(time_peaks_s1, -combined_2peaks_trailing[time_peaks_s1], "x")    
    ax.set_xlabel('sample')
    ax.set_ylabel('wavelength shift (nm)')
    ax.title.set_text('filtered combined L+R')
    ax.vlines(x=time_peaks_s1, ymin=-combined_2peaks_trailing[time_peaks_s1] - peak_prop_s1["prominences"],
        ymax = -combined_2peaks_trailing[time_peaks_s1])    
    ax.hlines(y=peak_prop_s1["width_heights"], xmin=peak_prop_s1["left_ips"],
                xmax=peak_prop_s1["right_ips"])
    
    ax.plot(x_values_wvt_arr, -combined_2peaks_leading, linewidth=2, label = 'leading')
    ax.plot(time_peaks_s2, -combined_2peaks_leading[time_peaks_s2], ".")
    ax.vlines(x=time_peaks_s2, ymin=-combined_2peaks_leading[time_peaks_s2] - peak_prop_s2["prominences"],
        ymax = -combined_2peaks_leading[time_peaks_s2])
    ax.hlines(y=peak_prop_s2["width_heights"], xmin=peak_prop_s2["left_ips"],
                xmax=peak_prop_s2["right_ips"])
    leg = ax.legend()
    
    tempDf = pd.DataFrame(data = [peak_prop_s1["peak_heights"]], 
                          columns=['steer_trailing_height','drive_trailing_height','trailer_trailing_height'])
    tempDf[['steer_trailing_width','drive_trailing_width','trailer_trailing_width']] = peak_prop_s1["widths"]
    tempDf[['steer_trailing_prominence','drive_trailing_prominence','trailer_trailing_prominence']] = peak_prop_s1["prominences"]
    tempDf[['steer_leading_height','drive_leading_height','trailer_leading_height']] = peak_prop_s2["peak_heights"]
    tempDf[['steer_leading_width','drive_leading_width','trailer_leading_width']] = peak_prop_s2["widths"]
    tempDf[['steer_leading_prominence','drive_leading_prominence','trailer_leading_prominence']] = peak_prop_s2["prominences"]
    tempDf[['steer_weight','drive_weight','trailer_weight']]=np.array([5.78,17.16,19.64])
    speed = 2.5/(signal.correlate(combined_2peaks_leading, combined_2peaks_trailing).argmax()-signal_length)*200*3.6
    bases = np.unique(np.concatenate((peak_prop_s2['right_bases'],peak_prop_s2['left_bases'])))
    tempDf['steer_leading_area'] = -np.trapz(combined_2peaks_leading[bases[0]:bases[1]])
    tempDf['drive_leading_area'] = -np.trapz(combined_2peaks_leading[bases[1]:bases[2]])
    tempDf['trailer_leading_area'] = -np.trapz(combined_2peaks_leading[bases[2]:bases[3]])
    bases = np.unique(np.concatenate((peak_prop_s1['right_bases'],peak_prop_s1['left_bases'])))
    tempDf['steer_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[0]:bases[1]])
    tempDf['drive_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[1]:bases[2]])
    tempDf['trailer_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[2]:bases[3]])
    tempDf[['speed','type']]=np.array([speed,2])
    df_features=pd.concat([df_features,tempDf])
    
    
# Van
figsize = (15, 1.5*6)
fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
axes = [ item for sublist in axes for item in sublist ] # flatten list

for i, df in enumerate([lane2run1van, lane2run2van, lane2run3van]):
    ax=axes[i]
    signal_raw_trailing_left = df.iloc[:,-8].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_trailing_right = df.iloc[:,-3].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_left = df.iloc[:,32].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_right = df.iloc[:,37].interpolate(method='linear', axis=0).ffill().bfill()
    combined_2peaks_trailing = (signal_raw_trailing_left+signal_raw_trailing_right)/2
    combined_2peaks_leading = (signal_raw_leading_left+signal_raw_leading_right)/2
    signal_length = len(signal_raw_trailing_left)
    # range of scales to perform the transform
    scales = scg.periods2scales( np.arange(1, 5,0.01) )
    x_values_wvt_arr = range(0,signal_length,1)
    ax.plot(x_values_wvt_arr, combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(x_values_wvt_arr, combined_2peaks_leading, linewidth=2, label = 'leading')
    ax.set_xlabel('sample')
    ax.set_ylabel('wavelength shift (nm)')
    ax.title.set_text('raw combined L+R')
    leg = ax.legend()
    
    ax=axes[i+3]
    if i==2:
        combined_2peaks_trailing = signal.detrend(signal.filtfilt(b, a, combined_2peaks_trailing))
        combined_2peaks_leading = signal.detrend(signal.filtfilt(b, a, combined_2peaks_leading))
    else:    
        combined_2peaks_trailing = signal.filtfilt(b, a, combined_2peaks_trailing)
        combined_2peaks_leading = signal.filtfilt(b, a, combined_2peaks_leading)
    time_peaks_s1, peak_prop_s1 = signal.find_peaks(-combined_2peaks_trailing, distance=3, prominence =0.0002, width=0.00003, height=0.00001)
    time_peaks_s2, peak_prop_s2 = signal.find_peaks(-combined_2peaks_leading, distance=3, prominence =0.0002, width=0.00003, height=0.00001)
    ax.plot(x_values_wvt_arr, -combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(time_peaks_s1, -combined_2peaks_trailing[time_peaks_s1], "x")    
    ax.set_xlabel('sample')
    ax.set_ylabel('wavelength shift (nm)')
    ax.title.set_text('filtered combined L+R')
    ax.vlines(x=time_peaks_s1, ymin=-combined_2peaks_trailing[time_peaks_s1] - peak_prop_s1["prominences"],
        ymax = -combined_2peaks_trailing[time_peaks_s1])    
    ax.hlines(y=peak_prop_s1["width_heights"], xmin=peak_prop_s1["left_ips"],
                xmax=peak_prop_s1["right_ips"])
    
    ax.plot(x_values_wvt_arr, -combined_2peaks_leading, linewidth=2, label = 'leading')
    ax.plot(time_peaks_s2, -combined_2peaks_leading[time_peaks_s2], ".")
    ax.vlines(x=time_peaks_s2, ymin=-combined_2peaks_leading[time_peaks_s2] - peak_prop_s2["prominences"],
        ymax = -combined_2peaks_leading[time_peaks_s2])
    ax.hlines(y=peak_prop_s2["width_heights"], xmin=peak_prop_s2["left_ips"],
                xmax=peak_prop_s2["right_ips"])
    leg = ax.legend()

    tempDf = pd.DataFrame(data = [peak_prop_s1["peak_heights"]], 
                          columns=['steer_trailing_height','drive_trailing_height'])
    tempDf[['steer_trailing_width','drive_trailing_width']] = peak_prop_s1["widths"]
    tempDf[['steer_trailing_prominence','drive_trailing_prominence']] = peak_prop_s1["prominences"]
    tempDf[['steer_leading_height','drive_leading_height']] = peak_prop_s2["peak_heights"]
    tempDf[['steer_leading_width','drive_leading_width']] = peak_prop_s2["widths"]
    tempDf[['steer_leading_prominence','drive_leading_prominence']] = peak_prop_s2["prominences"]
    tempDf[['steer_weight','drive_weight','trailer_weight']]=np.array([1.32,1.1,0])
    speed = 2.5/(signal.correlate(combined_2peaks_leading, combined_2peaks_trailing).argmax()-signal_length)*200*3.6
    bases = np.unique(np.concatenate((peak_prop_s2['right_bases'],peak_prop_s2['left_bases'])))
    tempDf['steer_leading_area'] = -np.trapz(combined_2peaks_leading[bases[0]:bases[1]])
    tempDf['drive_leading_area'] = -np.trapz(combined_2peaks_leading[bases[1]:bases[2]])
    bases = np.unique(np.concatenate((peak_prop_s1['right_bases'],peak_prop_s1['left_bases'])))
    tempDf['steer_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[0]:bases[1]])
    tempDf['drive_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[1]:bases[2]])
    # tempDf[['trailer_leading_area','trailer_trailing_area']] = np.array([0,0])
    tempDf[['speed','type']]=np.array([speed,3])
    df_features=pd.concat([df_features,tempDf])


# Lane5
figsize = (15, 1.5*6)
fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
axes = [ item for sublist in axes for item in sublist ] # flatten list

for i, df in enumerate([lane5run3rigid, lane5run3semi, lane5run3van]):
    ax=axes[i]
    signal_raw_trailing_left = df.iloc[:,2].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_trailing_right = df.iloc[:,8].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_left = df.iloc[:,43].interpolate(method='linear', axis=0).ffill().bfill()
    signal_raw_leading_right = df.iloc[:,47].interpolate(method='linear', axis=0).ffill().bfill()
    combined_2peaks_trailing = (signal_raw_trailing_left+signal_raw_trailing_right)/2
    combined_2peaks_leading = (signal_raw_leading_left+signal_raw_leading_right)/2
    signal_length = len(signal_raw_trailing_left)
    # range of scales to perform the transform
    scales = scg.periods2scales( np.arange(1, 5,0.01) )
    x_values_wvt_arr = range(0,signal_length,1)
    ax.plot(x_values_wvt_arr, combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(x_values_wvt_arr, combined_2peaks_leading, linewidth=2, label = 'leading')
    ax.set_xlabel('sample')
    ax.set_ylabel('wavelength shift (nm)')
    ax.title.set_text('raw combined L+R')
    leg = ax.legend()
    
    ax=axes[i+3]
    if i==2:
        combined_2peaks_trailing = signal.detrend(signal.filtfilt(b, a, combined_2peaks_trailing))
        combined_2peaks_leading = signal.detrend(signal.filtfilt(b, a, combined_2peaks_leading))
    else:    
        combined_2peaks_trailing = signal.filtfilt(b, a, combined_2peaks_trailing)
        combined_2peaks_leading = signal.filtfilt(b, a, combined_2peaks_leading)
    time_peaks_s1, peak_prop_s1 = signal.find_peaks(-combined_2peaks_trailing, distance=3, prominence =0.0002, width=0.00003, height=0.00001)
    time_peaks_s2, peak_prop_s2 = signal.find_peaks(-combined_2peaks_leading, distance=3, prominence =0.0002, width=0.00003, height=0.00001)
    ax.plot(x_values_wvt_arr, -combined_2peaks_trailing, linewidth=2, label = 'trailing')
    ax.plot(time_peaks_s1, -combined_2peaks_trailing[time_peaks_s1], "x")    
    ax.set_xlabel('sample')
    ax.set_ylabel('wavelength shift (nm)')
    ax.title.set_text('filtered combined L+R')
    ax.vlines(x=time_peaks_s1, ymin=-combined_2peaks_trailing[time_peaks_s1] - peak_prop_s1["prominences"],
        ymax = -combined_2peaks_trailing[time_peaks_s1])    
    ax.hlines(y=peak_prop_s1["width_heights"], xmin=peak_prop_s1["left_ips"],
                xmax=peak_prop_s1["right_ips"])
    
    ax.plot(x_values_wvt_arr, -combined_2peaks_leading, linewidth=2, label = 'leading')
    ax.plot(time_peaks_s2, -combined_2peaks_leading[time_peaks_s2], ".")
    ax.vlines(x=time_peaks_s2, ymin=-combined_2peaks_leading[time_peaks_s2] - peak_prop_s2["prominences"],
        ymax = -combined_2peaks_leading[time_peaks_s2])
    ax.hlines(y=peak_prop_s2["width_heights"], xmin=peak_prop_s2["left_ips"],
                xmax=peak_prop_s2["right_ips"])
    leg = ax.legend()

    if i == 1:
        tempDf = pd.DataFrame(data = [peak_prop_s1["peak_heights"]], 
                          columns=['steer_trailing_height','drive_trailing_height','trailer_trailing_height'])
        tempDf[['steer_trailing_width','drive_trailing_width','trailer_trailing_width']] = peak_prop_s1["widths"]
        tempDf[['steer_trailing_prominence','drive_trailing_prominence','trailer_trailing_prominence']] = peak_prop_s1["prominences"]
        tempDf[['steer_leading_height','drive_leading_height','trailer_leading_height']] = peak_prop_s2["peak_heights"]
        tempDf[['steer_leading_width','drive_leading_width','trailer_leading_width']] = peak_prop_s2["widths"]
        tempDf[['steer_leading_prominence','drive_leading_prominence','trailer_leading_prominence']] = peak_prop_s2["prominences"]
        tempDf[['steer_weight','drive_weight','trailer_weight']]=np.array([5.78,17.16,19.64])
        speed = 2.5/(signal.correlate(combined_2peaks_leading, combined_2peaks_trailing).argmax()-signal_length)*200*3.6
        bases = np.unique(np.concatenate((peak_prop_s2['right_bases'],peak_prop_s2['left_bases'])))
        tempDf['steer_leading_area'] = -np.trapz(combined_2peaks_leading[bases[0]:bases[1]])
        tempDf['drive_leading_area'] = -np.trapz(combined_2peaks_leading[bases[1]:bases[2]])
        tempDf['trailer_leading_area'] = -np.trapz(combined_2peaks_leading[bases[2]:bases[3]])
        bases = np.unique(np.concatenate((peak_prop_s1['right_bases'],peak_prop_s1['left_bases'])))
        tempDf['steer_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[0]:bases[1]])
        tempDf['drive_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[1]:bases[2]])
        tempDf['trailer_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[2]:bases[3]])
        tempDf[['speed','type']]=np.array([speed,2])
    else:
        tempDf = pd.DataFrame(data = [peak_prop_s1["peak_heights"]], 
                              columns=['steer_trailing_height','drive_trailing_height'])
        tempDf[['steer_trailing_width','drive_trailing_width']] = peak_prop_s1["widths"]
        tempDf[['steer_trailing_prominence','drive_trailing_prominence']] = peak_prop_s1["prominences"]
        tempDf[['steer_leading_height','drive_leading_height']] = peak_prop_s2["peak_heights"]
        tempDf[['steer_leading_width','drive_leading_width']] = peak_prop_s2["widths"]
        tempDf[['steer_leading_prominence','drive_leading_prominence']] = peak_prop_s2["prominences"]
        tempDf[['steer_weight','drive_weight','trailer_weight']]=np.array([1.32,1.1,0])
        speed = 2.5/(signal.correlate(combined_2peaks_leading, combined_2peaks_trailing).argmax()-signal_length)*200*3.6
        bases = np.unique(np.concatenate((peak_prop_s2['right_bases'],peak_prop_s2['left_bases'])))
        tempDf['steer_leading_area'] = -np.trapz(combined_2peaks_leading[bases[0]:bases[1]])
        tempDf['drive_leading_area'] = -np.trapz(combined_2peaks_leading[bases[1]:bases[2]])
        bases = np.unique(np.concatenate((peak_prop_s1['right_bases'],peak_prop_s1['left_bases'])))
        tempDf['steer_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[0]:bases[1]])
        tempDf['drive_trailing_area'] = -np.trapz(combined_2peaks_trailing[bases[1]:bases[2]])
        # tempDf[['trailer_leading_area','trailer_trailing_area']] = np.array([0,0])
    if i == 0:
        tempDf[['speed','type']]=np.array([speed,1])
    elif i == 2:
        tempDf[['speed','type']]=np.array([speed,3])
    
    df_features=pd.concat([df_features,tempDf])



        
#%% Analyze the features
df_features = df_features.abs()
# calculate the correlation matrix
corr = df_features.corr()

# plot the heatmap
fig, ax = plt.subplots(figsize=(16,9))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, ax=ax)
plt.xticks(rotation=90) 
plt.tight_layout()

df_rigid = df_features.loc[df_features['type'] == 1]

# set width of bar
barWidth = 0.25
r1 = np.arange(2)
r2 = np.arange(2)

figsize = (8, 6)
fig, axes = plt.subplots(2, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
for i in range(4):
    run1 = [df_rigid.steer_trailing_area.iloc[i], df_rigid.steer_leading_area.iloc[i]]
    r1 = [x + barWidth for x in r1]
    axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    # Add xticks on the middle of the group bars
    axes[0].set_xlabel('Rigid steer', fontweight='bold')
    axes[0].set_ylabel('AUC')
    axes[0].set_xticks([r + barWidth*2 for r in range(2)])
    axes[0].set_xticklabels(['trailing', 'leading'])
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0,0.4)
    
    run2 = [df_rigid.drive_trailing_area.iloc[i], df_rigid.drive_leading_area.iloc[i]]
    r2 = [x + barWidth for x in r2]
    axes[1].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[1].set_xlabel('Rigid drive', fontweight='bold')
    axes[1].set_ylabel('AUC')
    axes[1].set_xticks([r + barWidth*2 for r in range(2)])
    axes[1].set_xticklabels(['trailing', 'leading'])
    axes[1].legend(loc='upper left')
    axes[1].set_ylim(0,0.4)
plt.show()

# set width of bar
barWidth = 0.25
r1 = np.arange(2)
r2 = np.arange(2)
figsize = (8, 6)
fig, axes = plt.subplots(2, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
for i in range(4):
    run1 = [df_rigid.steer_trailing_height.iloc[i], df_rigid.steer_leading_height.iloc[i]]
    r1 = [x + barWidth for x in r1]
    axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    # Add xticks on the middle of the group bars
    axes[0].set_xlabel('Rigid steer', fontweight='bold')
    axes[0].set_ylabel('Peak height (WLS: nm)')
    axes[0].set_xticks([r + barWidth*2 for r in range(2)])
    axes[0].set_xticklabels(['trailing', 'leading'])
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0,0.015)
    
    run2 = [df_rigid.drive_trailing_height.iloc[i], df_rigid.drive_leading_height.iloc[i]]
    r2 = [x + barWidth for x in r2]
    axes[1].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[1].set_xlabel('Rigid drive', fontweight='bold')
    axes[1].set_ylabel('Peak height (WLS: nm)')
    axes[1].set_xticks([r + barWidth*2 for r in range(2)])
    axes[1].set_xticklabels(['trailing', 'leading'])
    axes[1].legend(loc='upper left')
    axes[1].set_ylim(0,0.015)
plt.show()


# Semi
df_semi = df_features.loc[df_features['type'] == 2]

# set width of bar
barWidth = 0.25
r1 = np.arange(2)
r2 = np.arange(2)
r3 = np.arange(2)

figsize = (8, 6)
fig, axes = plt.subplots(3, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.5, wspace=0.3 )
for i in range(4):
    run1 = [df_semi.steer_trailing_area.iloc[i], df_semi.steer_leading_area.iloc[i]]
    r1 = [x + barWidth for x in r1]
    axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    # Add xticks on the middle of the group bars
    axes[0].set_xlabel('semi steer', fontweight='bold')
    axes[0].set_ylabel('AUC')
    axes[0].set_xticks([r + barWidth*2 for r in range(2)])
    axes[0].set_xticklabels(['trailing', 'leading'])
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0,0.9)
    
    run2 = [df_semi.drive_trailing_area.iloc[i], df_semi.drive_leading_area.iloc[i]]
    r2 = [x + barWidth for x in r2]
    axes[1].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[1].set_xlabel('semi drive', fontweight='bold')
    axes[1].set_ylabel('AUC')
    axes[1].set_xticks([r + barWidth*2 for r in range(2)])
    axes[1].set_xticklabels(['trailing', 'leading'])
    axes[1].legend(loc='upper left')
    axes[1].set_ylim(0,0.9)
    
    run3 = [df_semi.trailer_trailing_area.iloc[i], df_semi.trailer_leading_area.iloc[i]]
    r3 = [x + barWidth for x in r3]
    axes[2].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[2].set_xlabel('semi drive', fontweight='bold')
    axes[2].set_ylabel('AUC')
    axes[2].set_xticks([r + barWidth*2 for r in range(2)])
    axes[2].set_xticklabels(['trailing', 'leading'])
    axes[2].legend(loc='upper left')
    axes[2].set_ylim(0,0.9)
plt.show()

# set width of bar
barWidth = 0.25
r1 = np.arange(2)
r2 = np.arange(2)
r3 = np.arange(2)

figsize = (8, 6)
fig, axes = plt.subplots(3, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.5, wspace=0.3 )
for i in range(4):
    run1 = [df_semi.steer_trailing_height.iloc[i], df_semi.steer_leading_height.iloc[i]]
    r1 = [x + barWidth for x in r1]
    axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    # Add xticks on the middle of the group bars
    axes[0].set_xlabel('semi steer', fontweight='bold')
    axes[0].set_ylabel('Peak height (WLS: nm)')
    axes[0].set_xticks([r + barWidth*2 for r in range(2)])
    axes[0].set_xticklabels(['trailing', 'leading'])
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0,0.025)
    
    run2 = [df_semi.drive_trailing_height.iloc[i], df_semi.drive_leading_height.iloc[i]]
    r2 = [x + barWidth for x in r2]
    axes[1].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[1].set_xlabel('semi drive', fontweight='bold')
    axes[1].set_ylabel('Peak height (WLS: nm)')
    axes[1].set_xticks([r + barWidth*2 for r in range(2)])
    axes[1].set_xticklabels(['trailing', 'leading'])
    axes[1].legend(loc='upper left')
    axes[1].set_ylim(0,0.025)
    
    run3 = [df_semi.trailer_trailing_height.iloc[i], df_semi.trailer_leading_height.iloc[i]]
    r3 = [x + barWidth for x in r3]
    axes[2].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[2].set_xlabel('semi trailer', fontweight='bold')
    axes[2].set_ylabel('Peak height (WLS: nm)')
    axes[2].set_xticks([r + barWidth*2 for r in range(2)])
    axes[2].set_xticklabels(['trailing', 'leading'])
    axes[2].legend(loc='upper left')
    axes[2].set_ylim(0,0.025)
plt.show()


# Van
df_van = df_features.loc[df_features['type'] == 3]

# set width of bar
barWidth = 0.25
r1 = np.arange(2)
r2 = np.arange(2)

figsize = (8, 6)
fig, axes = plt.subplots(2, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
for i in range(4):
    run1 = [df_van.steer_trailing_area.iloc[i], df_van.steer_leading_area.iloc[i]]
    r1 = [x + barWidth for x in r1]
    axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    # Add xticks on the middle of the group bars
    axes[0].set_xlabel('van steer', fontweight='bold')
    axes[0].set_ylabel('AUC')
    axes[0].set_xticks([r + barWidth*2 for r in range(2)])
    axes[0].set_xticklabels(['trailing', 'leading'])
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0,0.2)
    
    run2 = [df_van.drive_trailing_area.iloc[i], df_van.drive_leading_area.iloc[i]]
    r2 = [x + barWidth for x in r2]
    axes[1].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[1].set_xlabel('van drive', fontweight='bold')
    axes[1].set_ylabel('AUC')
    axes[1].set_xticks([r + barWidth*2 for r in range(2)])
    axes[1].set_xticklabels(['trailing', 'leading'])
    axes[1].legend(loc='upper left')
    axes[1].set_ylim(0,0.2)
plt.show()

# set width of bar
barWidth = 0.25
r1 = np.arange(2)
r2 = np.arange(2)
figsize = (8, 6)
fig, axes = plt.subplots(2, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
for i in range(4):
    run1 = [df_van.steer_trailing_height.iloc[i], df_van.steer_leading_height.iloc[i]]
    r1 = [x + barWidth for x in r1]
    axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    # Add xticks on the middle of the group bars
    axes[0].set_xlabel('van steer', fontweight='bold')
    axes[0].set_ylabel('Peak height (WLS: nm)')
    axes[0].set_xticks([r + barWidth*2 for r in range(2)])
    axes[0].set_xticklabels(['trailing', 'leading'])
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0,0.005)
    
    run2 = [df_van.drive_trailing_height.iloc[i], df_van.drive_leading_height.iloc[i]]
    r2 = [x + barWidth for x in r2]
    axes[1].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[1].set_xlabel('van drive', fontweight='bold')
    axes[1].set_ylabel('Peak height (WLS: nm)')
    axes[1].set_xticks([r + barWidth*2 for r in range(2)])
    axes[1].set_xticklabels(['trailing', 'leading'])
    axes[1].legend(loc='upper left')
    axes[1].set_ylim(0,0.005)
plt.show()


#%% Look at the combined AUC and height realtionship with weight
df = df_features.copy()
df_features = df

df_weight_trailing = pd.concat((df_features['steer_trailing_area']/df_features['steer_weight'],
                               df_features['drive_trailing_area']/df_features['drive_weight'],
                               df_features['trailer_trailing_area']/df_features['trailer_weight']), axis=0)
df_weight_trailing.index = ['lane2run1rigid_steer','lane2run2rigid_steer','lane2run3rigid_steer',
                            'lane2run1semi_steer','lane2run2semi_steer','lane2run3semi_steer',
                            'lane2run1van_steer','lane2run2van_steer','lane2run3van_steer',
                            'lane5run3rigid_steer','lane5run3semi_steer','lane5run3van_steer',
                            'lane2run1rigid_drive','lane2run2rigid_drive','lane2run3rigid_drive',
                            'lane2run1semi_drive','lane2run2semi_drive','lane2run3semi_drive',
                            'lane5run3van_drive','lane5run3van_drive','lane5run3van_drive',
                            'lane5run3rigid_drive','lane5run3semi_drive','lane5run3van_drive',
                            'lane2run1rigid_trailer','lane2run2rigid_trailer','lane2run3rigid_trailer',
                            'lane2run1semi_trailer','lane2run2semi_trailer','lane2run3semi_trailer',
                            'lane2run1van_trailer','lane2run2van_trailer','lane2run3van_trailer',
                            'lane5run3rigid_trailer','lane5run3semi_trailer','lane5run3van_trailer']

df_weight_leading = pd.concat((df_features['steer_leading_area']/df_features['steer_weight'],
                               df_features['drive_leading_area']/df_features['drive_weight'],
                               df_features['trailer_leading_area']/df_features['trailer_weight']), axis=0)

df_weight_leading.index = ['lane2run1rigid_steer','lane2run2rigid_steer','lane2run3rigid_steer',
                            'lane2run1semi_steer','lane2run2semi_steer','lane2run3semi_steer',
                            'lane2run1van_steer','lane2run2van_steer','lane2run3van_steer',
                            'lane5run3rigid_steer','lane5run3semi_steer','lane5run3van_steer',
                            'lane2run1rigid_drive','lane2run2rigid_drive','lane2run3rigid_drive',
                            'lane2run1semi_drive','lane2run2semi_drive','lane2run3semi_drive',
                            'lane5run3van_drive','lane5run3van_drive','lane5run3van_drive',
                            'lane5run3rigid_drive','lane5run3semi_drive','lane5run3van_drive',
                            'lane2run1rigid_trailer','lane2run2rigid_trailer','lane2run3rigid_trailer',
                            'lane2run1semi_trailer','lane2run2semi_trailer','lane2run3semi_trailer',
                            'lane2run1van_trailer','lane2run2van_trailer','lane2run3van_trailer',
                            'lane5run3rigid_trailer','lane5run3semi_trailer','lane5run3van_trailer']
 

# look at the height related to weight
df_weight_trailing_mean = df_weight_trailing.mean()
df_weight_trailing_std = df_weight_trailing.std()

figsize = (8, 9)
fig, axes = plt.subplots(2, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.5, wspace=0.3 )
df_weight_trailing.plot(x='x', y='y', kind='bar', ax=axes[0])
axes[0].set_ylim(0,0.08)
axes[0].tick_params(axis='x', rotation=30)

df_weight_leading.plot(x='x', y='y', kind='bar')
axes[1].set_ylim(0,0.08)
axes[1].tick_params(axis='x', rotation=30)
plt.suptitle('AUC devide by control weight')
plt.tight_layout()


df_height_trailing = pd.concat((df_features['steer_trailing_height']/df_features['steer_weight'],
                               df_features['drive_trailing_height']/df_features['drive_weight'],
                               df_features['trailer_trailing_height']/df_features['trailer_weight']), axis=0)
df_height_trailing.index = ['lane2run1rigid_steer','lane2run2rigid_steer','lane2run3rigid_steer',
                            'lane2run1semi_steer','lane2run2semi_steer','lane2run3semi_steer',
                            'lane2run1van_steer','lane2run2van_steer','lane2run3van_steer',
                            'lane5run3rigid_steer','lane5run3semi_steer','lane5run3van_steer',
                            'lane2run1rigid_drive','lane2run2rigid_drive','lane2run3rigid_drive',
                            'lane2run1semi_drive','lane2run2semi_drive','lane2run3semi_drive',
                            'lane5run3van_drive','lane5run3van_drive','lane5run3van_drive',
                            'lane5run3rigid_drive','lane5run3semi_drive','lane5run3van_drive',
                            'lane2run1rigid_trailer','lane2run2rigid_trailer','lane2run3rigid_trailer',
                            'lane2run1semi_trailer','lane2run2semi_trailer','lane2run3semi_trailer',
                            'lane2run1van_trailer','lane2run2van_trailer','lane2run3van_trailer',
                            'lane5run3rigid_trailer','lane5run3semi_trailer','lane5run3van_trailer']

df_height_leading = pd.concat((df_features['steer_leading_height']/df_features['steer_weight'],
                               df_features['drive_leading_height']/df_features['drive_weight'],
                               df_features['trailer_leading_height']/df_features['trailer_weight']), axis=0)

df_height_leading.index = ['lane2run1rigid_steer','lane2run2rigid_steer','lane2run3rigid_steer',
                            'lane2run1semi_steer','lane2run2semi_steer','lane2run3semi_steer',
                            'lane2run1van_steer','lane2run2van_steer','lane2run3van_steer',
                            'lane5run3rigid_steer','lane5run3semi_steer','lane5run3van_steer',
                            'lane2run1rigid_drive','lane2run2rigid_drive','lane2run3rigid_drive',
                            'lane2run1semi_drive','lane2run2semi_drive','lane2run3semi_drive',
                            'lane5run3van_drive','lane5run3van_drive','lane5run3van_drive',
                            'lane5run3rigid_drive','lane5run3semi_drive','lane5run3van_drive',
                            'lane2run1rigid_trailer','lane2run2rigid_trailer','lane2run3rigid_trailer',
                            'lane2run1semi_trailer','lane2run2semi_trailer','lane2run3semi_trailer',
                            'lane2run1van_trailer','lane2run2van_trailer','lane2run3van_trailer',
                            'lane5run3rigid_trailer','lane5run3semi_trailer','lane5run3van_trailer']

df_height_trailing_mean = df_height_trailing.mean()
df_height_trailing_std = df_height_trailing.std()

# look at the height related to weight
figsize = (8, 9)
fig, axes = plt.subplots(2, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.5, wspace=0.3 )
df_height_trailing.plot(x='x', y='y', kind='bar', ax=axes[0])
axes[0].set_ylim(0,0.003)
axes[0].tick_params(axis='x', rotation=30)

df_height_leading.plot(x='x', y='y', kind='bar')
axes[1].set_ylim(0,0.003)
axes[1].tick_params(axis='x', rotation=30)
plt.suptitle('height devide by control weight')
plt.tight_layout()


df_auc = pd.concat([df_weight_trailing,df_weight_leading],axis=0)
df_height = pd.concat([df_height_trailing,df_height_leading],axis=0)

# density distribution
df_auc.plot.kde(label='AUC/weight distribution')
df_height.plot.kde(label='height/weight distribution')

plt.figure()
sns.distplot(df_auc, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'AUC/weight')
plt.legend()
plt.ylabel('Density')

# plt.figure()
sns.distplot(df_height, hist = False, kde = True,
                 kde_kws = {'linewidth': 3},
                 label = 'height/weight')
plt.legend()
plt.ylabel('Density')

kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

plt.figure()
sns.distplot(df_auc, **kwargs, color='g', label='AUC')
sns.distplot(df_height, **kwargs, color='b', label='Height')
plt.title('Historgram and density of features divide by control axle group weights')
plt.ylim(0,400)
plt.legend()

# compare the normalized deistribution
df_auc = sklearn.preprocessing.minmax_scale(df_auc, feature_range=(0, 1), axis=0, copy=True)
df_height = sklearn.preprocessing.minmax_scale(df_height, feature_range=(0, 1), axis=0, copy=True)


#%%
# rigid
df_rigid = df_features.loc[df_features['type'] == 1]

# set width of bar
barWidth = 0.2
r1 = np.arange(3)
r2 = np.arange(3)
r3 = np.arange(3)

figsize = (8, 8)
fig, axes = plt.subplots(3, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
for i in range(4):
    combine_rigid = (df_rigid.steer_trailing_area+df_rigid.steer_leading_area)
    combine_semi = (df_semi.steer_trailing_area+df_semi.steer_leading_area)
    combine_van = (df_van.steer_trailing_area+df_van.steer_leading_area)
    run1 = [combine_rigid.iloc[i], combine_semi.iloc[i], combine_van.iloc[i]]
    r1 = [x + barWidth for x in r1]
    if i == 3:
        axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Lane 5 Run '+str(i+1))
    else:
        axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Lane 2 Run '+str(i+1))
    # Add xticks on the middle of the group bars
    axes[0].set_xlabel('Steer', fontweight='bold')
    axes[0].set_ylabel('AUC')
    axes[0].set_xticks([r + barWidth*2 for r in range(3)])
    axes[0].set_xticklabels(['rigid', 'semi', 'van'])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          ncol=2, fancybox=True, shadow=True)
    axes[0].set_ylim([0,1.4])

    combine_rigid = (df_rigid.drive_trailing_area+df_rigid.drive_leading_area)
    combine_semi = (df_semi.drive_trailing_area+df_semi.drive_leading_area)
    combine_van = (df_van.drive_trailing_area+df_van.drive_leading_area)
    run2 = [combine_rigid.iloc[i], combine_semi.iloc[i], combine_van.iloc[i]]
    r2 = [x + barWidth for x in r2]
    axes[1].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[1].set_xlabel('Drive', fontweight='bold')
    axes[1].set_ylabel('AUC')
    axes[1].set_xticks([r + barWidth*2 for r in range(3)])
    axes[1].set_xticklabels(['rigid', 'semi', 'van'])
    axes[1].set_ylim([0,1.4])
    
    combine_rigid = (df_rigid.trailer_trailing_area+df_rigid.trailer_leading_area)
    combine_semi = (df_semi.trailer_trailing_area+df_semi.trailer_leading_area)
    combine_van = (df_van.trailer_trailing_area+df_van.trailer_leading_area)
    run3 = [0, combine_semi.iloc[i], 0]
    r3 = [x + barWidth for x in r3]
    axes[2].bar(r3, run3, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[2].set_xlabel('Trailer', fontweight='bold')
    axes[2].set_ylabel('AUC')
    axes[2].set_xticks([r + barWidth*2 for r in range(3)])
    axes[2].set_xticklabels(['rigid', 'semi', 'van'])
    axes[2].set_ylim([0,1.4])
plt.show()

# set width of bar
barWidth = 0.2
r1 = np.arange(3)
r2 = np.arange(3)
r3 = np.arange(3)

figsize = (8, 8)
fig, axes = plt.subplots(3, 1, figsize=figsize)
plt.subplots_adjust(hspace=0.3, wspace=0.3 )
for i in range(4):
    combine_rigid = (df_rigid.steer_trailing_height+df_rigid.steer_leading_height)
    combine_semi = (df_semi.steer_trailing_height+df_semi.steer_leading_height)
    combine_van = (df_van.steer_trailing_height+df_van.steer_leading_height)
    run1 = [combine_rigid.iloc[i], combine_semi.iloc[i], combine_van.iloc[i]]
    r1 = [x + barWidth for x in r1]
    if i == 3:
        axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Lane 5 Run '+str(i+1))
    else:
        axes[0].bar(r1, run1, width=barWidth, edgecolor='white', label='Lane 2 Run '+str(i+1))
    # Add xticks on the middle of the group bars
    axes[0].set_xlabel('Steer', fontweight='bold')
    axes[0].set_ylabel('Height')
    axes[0].set_xticks([r + barWidth*2 for r in range(3)])
    axes[0].set_xticklabels(['rigid', 'semi', 'van'])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          ncol=2, fancybox=True, shadow=True)
    axes[0].set_ylim([0,0.04])
    
    combine_rigid = (df_rigid.drive_trailing_height+df_rigid.drive_leading_height)
    combine_semi = (df_semi.drive_trailing_height+df_semi.drive_leading_height)
    combine_van = (df_van.drive_trailing_height+df_van.drive_leading_height)
    run2 = [combine_rigid.iloc[i], combine_semi.iloc[i], combine_van.iloc[i]]
    r2 = [x + barWidth for x in r2]
    axes[1].bar(r2, run2, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[1].set_xlabel('Drive', fontweight='bold')
    axes[1].set_ylabel('Height')
    axes[1].set_xticks([r + barWidth*2 for r in range(3)])
    axes[1].set_xticklabels(['rigid', 'semi', 'van'])
    axes[1].set_ylim([0,0.04])
    
    combine_rigid = (df_rigid.trailer_trailing_height+df_rigid.trailer_leading_height)
    combine_semi = (df_semi.trailer_trailing_height+df_semi.trailer_leading_height)
    combine_van = (df_van.trailer_trailing_height+df_van.trailer_leading_height)
    run3 = [0, combine_semi.iloc[i], 0]
    r3 = [x + barWidth for x in r3]
    axes[2].bar(r3, run3, width=barWidth, edgecolor='white', label='Run '+str(i+1))
    axes[2].set_xlabel('Trailer', fontweight='bold')
    axes[2].set_ylabel('Height')
    axes[2].set_xticks([r + barWidth*2 for r in range(3)])
    axes[2].set_xticklabels(['rigid', 'semi', 'van'])
    axes[2].set_ylim([0,0.04])
plt.show()
