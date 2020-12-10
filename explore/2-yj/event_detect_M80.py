from datetime import timedelta, datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from util import imagepers
from scipy.signal import butter, lfilter, freqz
import math

from pydriveeasy.visualize import plot_heatmap, PersistentTopology, Heat3D
from util.util import helper
        
    
#%% Read data and sidentify wrong sensors
fiber1 = helper.load_from_npz('M80_DriveEasy/wav/wav_20201202_024632_F01_UTC.npz')
fiber2 = helper.load_from_npz('M80_DriveEasy/wav/wav_20201202_024632_F02_UTC.npz')
fiber3 = helper.load_from_npz('M80_DriveEasy/wav/wav_20201202_024632_F03_UTC.npz')
fiber4 = helper.load_from_npz('M80_DriveEasy/wav/wav_20201202_024632_F04_UTC.npz')

ch1_base = np.array([1511.4551,1514.8723,1518.5089,1521.9215,1525.2612,1528.5006,1531.6403,1534.5469,1539.3361,1540.7187,1544.0044,1547.2896,1550.5481,1553.925,1557.2499,1560.3572,1563.5336,1566.7323,1569.87,1573.1298,1576.1335,1578.9739,1581.8887,1584.8264,1588.6771])
ch2_base = np.array([1511.5165,1515.1094,1518.7858,1522.1046,1525.42,1528.7603,1531.9319,1534.9578,1539.7893,1541.2045,1544.5333,1547.7975,1551.1421,1554.2081,1557.3563,1560.668,1563.9662,1567.2091,1570.3817,1573.6971,1576.7531,1579.7942,1582.643,1585.076,1588.5071])
ch3_base = np.array([1511.5145,1515.067,1518.3971,1521.6411,1524.8749,1528.1523,1531.4237,1534.5928,1538.9817,1540.9666,1544.29,1547.6348,1550.8978,1554.1719,1557.4568,1560.7444,1563.8342,1567.0056,1570.0883,1573.2153,1576.4351,1579.5439,1582.5308,1585.3574,1589.3844])
ch4_base = np.array([1512.189,1514.8191,1518.2963,1521.7609,1524.9905,1528.1858,1531.401,1534.5953,1538.8943,1541.0115,1544.1846,1547.4498,1550.6222,1553.8363,1557.1041,1560.2871,1563.6326,1566.8469,1570.0469,1573.1624,1576.3771,1579.4347,1582.4053,1585.5077,1589.1851])


fibers = [fiber1, fiber2, fiber3, fiber4]

for fiber_id, fiber in enumerate(fibers):
    fibers[fiber_id] = helper.wavelength_check(fiber_id,fiber)
    fibers[fiber_id] = fibers[fiber_id].add_prefix('ch'+str(fiber_id+1)+'_')
    # fiber.plot()
fiber1 = fibers[0]
fiber2 = fibers[1]
fiber3 = fibers[2]
fiber4 = fibers[3]

# # check fiber
# check = fiber1.index.diff().dt.total_seconds()
# check = fiber2.index.to_series().diff().dt.total_seconds()
# check3 = fiber3.index.to_series().diff().dt.total_seconds()
# check4 = fiber4.index.to_series().diff().dt.total_seconds()

# check1 = fiber3.index.to_series().reset_index(drop=True)
# check2 = fiber4.index.to_series().reset_index(drop=True)
# check=(check2-check1).dt.total_seconds()

#%% Put sensors in lanes, firs 25 sensors are leading
lane_west = pd.concat([fiber4.iloc[:,0:16],fiber2.iloc[:,9:25]], axis=1)
lane_east = pd.concat([fiber3.iloc[:,0:16],fiber1.iloc[:,9:25]], axis=1)

lane_west0 = np.concatenate([ch4_base[0:16],ch2_base[9:25]], axis=0)
lane_east0 = np.concatenate([ch3_base[0:16],ch1_base[9:25]], axis=0)

def speed_estimate(fiber1, fiber2):    
    # time difference from peak values 
    time_diff = fiber1.idxmax().values-fiber2.idxmax().values
    time_diff = time_diff/np.timedelta64(1, 's')
    
    # distance (2.5 m) divide by the time difference, 2.23694 is tranferring m/s to mph
    velocity = 2.5/time_diff*3.6
    
    # threshold the abnormal data beyond vehicle speed limit.
    velocity[(-5<velocity) & (velocity<5)] = np.nan
    velocity[(velocity>70) | (velocity<-70)] = np.nan
    velocity_median = np.nanmedian(velocity)
    return velocity_median


# Function to find distance 
def shortest_distance(x1, y1, a, b, c):        
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b)) 
    return d
    
    
def event_detection(window_west, window_east, frame, visualize=False):   
    # Step 1: threshold the window see if wavelenth shift larger than threshold
    if window_west[window_west.abs()>0.002].any().any():  
        
    # Step 2: threshold through homology plot
    # refer to the plot, set visualize=True
        im_w = window_west.fillna(0).values
        im_w = im_w * (255 / np.max(im_w))
        g0_w = imagepers.persistence(im_w)  
        
        pks_w = []
        for i, homclass in enumerate(g0_w[0:20]):
            p_birth, bl, pers, p_death = homclass
            if shortest_distance(bl,bl-pers,1,-1,0)>20:
                pks_w.append([p_birth, bl, pers, p_death])
            
        im_e = window_east.fillna(0).values
        im_e = im_e * (255 / np.max(im_e))
        g0_e = imagepers.persistence(im_e)  
        
        pks_e = []
        for i, homclass in enumerate(g0_e[0:20]):
            p_birth, bl, pers, p_death = homclass
            if shortest_distance(bl,bl-pers,1,-1,0)>20:
                pks_e.append([p_birth, bl, pers, p_death])           
        
        if len(pks_e)>0 | len(pks_w)>0:
            if visualize == True:
                Heat3D(im_w, pks_w, g0_w, window_west, im_e, pks_e, g0_e, window_east, frame)
            return 1, pks_w, pks_e
        else:
            return 0, [], []
    else:
        return 0, [], []

        
def directionLocator(window_west, window_east):
    """
    Identify the driving direction through max response sensor IDs.
    Parameters
    ----------
    window_west : TYPE
        DESCRIPTION.
    window_east : TYPE
        DESCRIPTION.

    Returns
    -------
    direction : TYPE
        DESCRIPTION.

    """
    # max response sensor IDs in each column
    west_max = np.argmax(window_west.abs().values, axis=0)
    east_max = np.argmax(window_east.abs().values, axis=0)
    # most frequent sensor IDs
    west_max_frequent = np.bincount(west_max).argmax()
    west_max_frequent = np.bincount(east_max).argmax()
    direction, leading, trailing = None, [], []
    # determine the most frequent reponse sensor IDs locate in which direction.
    if (np.where(west_max == west_max_frequent)[0]<25).all():
        direction = 'west'
        leading = window_west.iloc[:,0:25]
        trailing = window_east.iloc[:,25::]
    elif (np.where(west_max == west_max_frequent)[0]>25).all():
        direction = 'east'      
        leading = window_east.iloc[:,0:25]
        trailing = window_west.iloc[:,25::]
    return direction, leading, trailing


def LaneLocator(window_west, window_east, pks_w, pks_e):
    lanes = np.array([0,0,0,0])
    leading_id, trailing_id = [], []
    for item in pks_w:
        if item[0][1]<8:
            lanes[0] += 1
        elif 11<=item[0][1]<16:
            lanes[1] += 1
        elif 21<=item[0][1]<24:
            lanes[2] += 1
        elif 32<=item[0][1]<=42:
            lanes[3] += 1
            
    for item in pks_e:
        if item[0][1]<8:
            lanes[0] += 1
        elif 11<=item[0][1]<16:
            lanes[1] += 1
        elif 21<=item[0][1]<24:
            lanes[2] += 1
        elif 32<=item[0][1]<=42:
            lanes[3] += 1
            
    # lane_id = str(5-lanes.argmax())
    # if lane_id == '5':
    #     leading = window_west.iloc[:,0:11]        
    #     trailing = window_east.iloc[:,0:11]
    # elif lane_id == '4':
    #     leading = window_west.iloc[:,11:21]
    #     trailing = window_east.iloc[:,11:21]
    # elif lane_id == '3':
    #     leading = window_east.iloc[:,21:32]
    #     trailing = window_west.iloc[:,21:32]
    # elif lane_id == '2':
    #     leading = window_east.iloc[:,32::]
    #     trailing = window_west.iloc[:,32::]
    
    leading_id = [item[0][1] for item in pks_w]
    trailing_id = [item[0][1] for item in pks_e]
    return lanes, leading_id, trailing_id 

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def ExtractEvents(i, window_west, window_east, leading_id, trailing_id, visualize=False):
    # merge two lists without duplicates
    combined_ids = trailing_id + list(set(leading_id)-set(trailing_id))
    combined_west = window_west.iloc[:,combined_ids].sum(axis = 1)
    combined_east = window_east.iloc[:,combined_ids].sum(axis = 1)

    # make sure the line1 and line2 sensor ids are the same from one vehicle.
    time_peaks1, peak_prop1 = signal.find_peaks(abs(combined_west), distance=5, height = 0.001, prominence =0.005, width=0.5)
    time_peaks2, peak_prop2 = signal.find_peaks(abs(combined_east), distance=5, height = 0.001, prominence =0.005, width=0.5)
    n_peaks = max(len(time_peaks1), len(time_peaks2))  
    
    # make sure the number of peaks are largern than 1
    if n_peaks > 1:   
        # # identify the peaks (for 'signal.find_peaks' parameter adjusting purposes)
        if visualize == True:
            plt.figure()
            plt.plot(combined_west)
            plt.plot(combined_east, color = 'r')
            plt.plot(time_peaks1, combined_west.iloc[time_peaks1], "x")
            plt.plot(time_peaks2, combined_east.iloc[time_peaks2], "x", color = 'r')
            plt.vlines(x=time_peaks1, ymin=combined_west.iloc[time_peaks1] - peak_prop1["prominences"],
                ymax = combined_west.iloc[time_peaks1], color = "C1")
            plt.hlines(y=peak_prop1["width_heights"], xmin=peak_prop1["left_ips"],
                        xmax=peak_prop1["right_ips"], color = "C1")
            plt.show()
        
        # preset window peaks properties
        window_widths = np.zeros((10,100))
        window_left = np.zeros((10,100))
        window_right = np.zeros((10,100))
        
        # select adjacent 20 windows for analysis
        for j in range(10):
            window_s_start = window_size*(i) - 500 + (200*j)
            window_s_end = window_size*(i+1) - 500 + (200*j)
            
            # smaller sliding windows near the window_i for analysis 
            window_s1 = lane_west.iloc[window_s_start:window_s_end, :] - lane_west.iloc[window_s_start, :].values.squeeze() # zero the window
            window_s2 = lane_east.iloc[window_s_start:window_s_end, :] - lane_east.iloc[window_s_start, :].values.squeeze() # zero the window
            
            # get the peaks in this sliding windows
            time_peaks_s1, peak_prop_s1 = signal.find_peaks(abs(window_s1.iloc[:,combined_ids].sum(axis = 1)), distance=5, height = 0.005, width=0.5)
            time_peaks_s2, peak_prop_s2 = signal.find_peaks(abs(window_s2.iloc[:,combined_ids].sum(axis = 1)), distance=5, height = 0.005, width=0.5)
            
            if min(len(peak_prop_s1["peak_heights"]),len(peak_prop_s2["peak_heights"]))>0:
                if peak_prop_s1["peak_heights"].max()>peak_prop_s2["peak_heights"].max():
                    time_peaks, peak_prop = time_peaks_s1, peak_prop_s1
                else:
                    time_peaks, peak_prop = time_peaks_s2, peak_prop_s2
                
                n_peaks_temp = len(time_peaks)
                window_widths[j,0:n_peaks_temp] = peak_prop["widths"]
                window_left[j,:n_peaks_temp] = peak_prop["left_bases"]
                window_right[j,:n_peaks_temp] = peak_prop["right_bases"]                    
            
        # Make a classification prediction with neighbors
        distances = list()
        for idx, train_row in enumerate(window_widths):
            dist = helper.euclidean_distance(peak_prop1["widths"], train_row)
            distances.append((train_row, dist, idx))
            distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        neighbors_idx = list()
        
        for ii in range(len(distances)):
            if distances[ii][1] < 10:
                neighbors.append(distances[ii][0])
                neighbors_idx.append(distances[ii][-1])
          
        event_start, event_end = 0, 0
        event_window = []
        
        if len(neighbors_idx) >= 2:
            event_left_index = np.argmin(neighbors_idx)
            event_left_window = neighbors_idx[event_left_index]
            event_start = window_left[event_left_window,0]
            event_start = int(window_size*(i) - 500 + (100*event_left_window) + event_start - 10)
            
            event_right_index = np.argmax(neighbors_idx)
            event_right_window = neighbors_idx[event_right_index]
            event_end = window_right[event_right_window,:].max()
            event_end = int(window_size*(i) - 500 + (100*event_right_window) + event_end + 10)
                                
            event_window = lane_east.iloc[event_start:event_end,:]-lane_east.iloc[event_start,:].values.squeeze()
            event_window.index=pd.to_datetime(event_window.index)
            
            # get event window timestamp
            t_start = str(lane_east.index[window_size*(i)])
            t_start = datetime.strptime(t_start, '%Y-%m-%d %H:%M:%S.%f')
            t_start_str = datetime.strftime(t_start, '%Y_%m%d_%H_%M_%S')
            
            # wavelength shift
            wav1 = lane_east.iloc[event_start:event_end,:]
            wav2 = lane_west.iloc[event_start:event_end,:]
            
            speed = speed_estimate(wav1, wav2)
            speed_dic.append([event_start, speed])    
            plot_heatmap(event_window, 'lane_west3_Event'+ str(count), t_start_str +', speed (km/h)'+str(speed))
            
            # save the events                        
            helper.save_events(count, t_start_str, wav1, wav2, 3, 4, leading_id, trailing_id)
            
        else:
            # get event window timestamp
            t_start = str(lane_east.index[window_size*(i)])
            t_start = datetime.strptime(t_start, '%Y-%m-%d %H:%M:%S.%f')
            t_start_str = datetime.strftime(t_start, '%Y%m%d_%H%M%S')
            
            # wavelength shift
            wav1 = lane_east.iloc[window_size*(i):window_size*(i+1),:]
            wav2 = lane_west.iloc[window_size*(i):window_size*(i+1),:]
                                
            # speed estimation using median
            speed = speed_estimate(wav1, wav2)
            speed_dic.append([window_size*(i), speed]) 
            
            event_window = lane_east.iloc[window_size*(i):window_size*(i+1),:] - lane_east.iloc[window_size*(i), :].values.squeeze() # zero the window
            
            # visualization                        
            plot_heatmap(event_window, 'lane_west3_Event'+ str(count), t_start_str +', '+"%.2f" %speed+'km/h')
            
            # save the events                        
            helper.save_events(count, t_start_str, wav1, wav2, 3, 4, leading_id, trailing_id)
        return speed

#%% moving window for lane_west:
fs = 200 # Hz, sampling frequency
window_size = int(1*fs) # 1 second window size
speed_dic = []
count = 1
results = []

for win_i in range(int(lane_west.shape[0]/window_size)): 

    # prepare the windows for analysis
    window_west = lane_west.iloc[window_size*(win_i):window_size*(win_i+1),:]
    window_west.iloc[abs(window_west['ch4_sensor1']-1512.35)<1e-2,:] = np.nan     
    window_west = window_west - window_west.iloc[0,:].values.squeeze()   
    window_west.reset_index(drop=True, inplace=True)
    # window_west[abs(window_west)>0.01]=np.nan
    
    start_index = window_west.index[0]
    end_index = window_west.index[-1]
    
    window_east = lane_east.iloc[window_size*(win_i):window_size*(win_i+1),:]
    window_east.iloc[abs(window_east['ch3_sensor4']-1522.67)<1e-2,:] = np.nan  
    window_east = window_east - window_east.iloc[0,:].values.squeeze()    
    window_east.reset_index(drop=True, inplace=True)
    # window_east[abs(window_east)>0.01]=np.nan

    # event detection   
    event_flag, pks_w, pks_e = event_detection(window_west, window_east, win_i, False)
        
    if event_flag == 1:
        # determine driving direction
        # direction, leading_ch, trailing_ch = directionLocator(window_west, window_east)
        lanes, leading_id, trailing_id  = LaneLocator(window_west, window_east, pks_w, pks_e)

        for lane_id, lane in enumerate(lanes):
            if lane_id == 0:
                leading = window_west.iloc[:,0:11]        
                trailing = window_east.iloc[:,0:11]
                lane_leading_id = [ind for ind in leading_id if ind <11]
                lane_trailing_id = [ind for ind in leading_id if ind <11]
            elif lane_id == 1:
                leading = window_west.iloc[:,11:21]        
                trailing = window_east.iloc[:,11:21]
                lane_leading_id = [ind for ind in leading_id if 11<= ind <21]
                lane_trailing_id = [ind for ind in leading_id if 11<= ind <21]
            elif lane_id == 2:
                leading = window_west.iloc[:,21:32]        
                trailing = window_east.iloc[:,21:32]
                lane_leading_id = [ind for ind in leading_id if 21<= ind <32]
                lane_trailing_id = [ind for ind in leading_id if 21<= ind <32]
                speed = ExtractEvents(win_i, window_west, window_east, lane_leading_id, lane_trailing_id)
                
            elif lane_id == 3:
                leading = window_west.iloc[:,32::]        
                trailing = window_east.iloc[:,32::]
                lane_leading_id = [ind for ind in leading_id if 32<= ind <42]
                lane_trailing_id = [ind for ind in leading_id if 32<= ind <42]
            
        
        ExtractEvents()
                
        count += 1
                
        results.append([lane_id, speed])
    else:
        event_detect = 0
        
fig, ax = plt.subplots()
for i in range(len(speed_dic)):
    plt.vlines(speed_dic[i][0], 0, speed_dic[i][1], color = 'b')
    plt.scatter(speed_dic[i][0], speed_dic[i][1], color = 'b')
ax.set_ylabel('speed (km/h)')
ax.set_ylim([-70, 70])
ax.set_xticks([])
ax.set_title('event counts: ' + str(i+1))
plt.show()