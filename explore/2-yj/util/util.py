import scipy
import copy
import math
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from scipy import signal
import scipy
from sklearn import mixture
import copy
from scipy.ndimage.interpolation import shift

# Not a class, just a bunch of useful functions.

class helper:
    def wavelength_check(fiber_id, fiber):
        temp = fiber.copy()
        # determine which sensing point is bad
        ch1_base = np.array([1511.4551,1514.8723,1518.5089,1521.9215,1525.2612,1528.5006,1531.6403,1534.5469,1539.3361,1540.7187,1544.0044,1547.2896,1550.5481,1553.925,1557.2499,1560.3572,1563.5336,1566.7323,1569.87,1573.1298,1576.1335,1578.9739,1581.8887,1584.8264,1588.6771])
        ch2_base = np.array([1511.5165,1515.1094,1518.7858,1522.1046,1525.42,1528.7603,1531.9319,1534.9578,1539.7893,1541.2045,1544.5333,1547.7975,1551.1421,1554.2081,1557.3563,1560.668,1563.9662,1567.2091,1570.3817,1573.6971,1576.7531,1579.7942,1582.643,1585.076,1588.5071])
        ch3_base = np.array([1511.5145,1515.067,1518.3971,1521.6411,1524.8749,1528.1523,1531.4237,1534.5928,1538.9817,1540.9666,1544.29,1547.6348,1550.8978,1554.1719,1557.4568,1560.7444,1563.8342,1567.0056,1570.0883,1573.2153,1576.4351,1579.5439,1582.5308,1585.3574,1589.3844])
        ch4_base = np.array([1512.189,1514.8191,1518.2963,1521.7609,1524.9905,1528.1858,1531.401,1534.5953,1538.8943,1541.0115,1544.1846,1547.4498,1550.6222,1553.8363,1557.1041,1560.2871,1563.6326,1566.8469,1570.0469,1573.1624,1576.3771,1579.4347,1582.4053,1585.5077,1589.1851])
        fiber_bases = [ch1_base, ch2_base, ch3_base, ch4_base]
        error_cols = []
        error_cols = [col for col in fiber.columns if (fiber[col]<1500).all()]
        
        if len(error_cols)==1:
            fiber.iloc[:,1::] = temp.iloc[:,0:24].values
            fiber['sensor1'] = np.nan
            print("channel "+str(fiber_id+1)+", sensor 1 has error")
        elif len(error_cols)==2:
            fiber.iloc[:,2::] = temp.iloc[:,0:23].values
            fiber[['sensor1','sensor2']] = np.nan
            print("channel "+str(fiber_id+1)+", sensor 1,2 has error")
        
        # threshold switching issue outlier
        fiber[abs(fiber.values-fiber_bases[fiber_id])>3] = np.nan
        
        # identify sampling issue outliers
        check = fiber.index.to_series().diff().dt.total_seconds()
        df = pd.DataFrame(index=(fiber[check.values>0.006].index-timedelta(seconds=0.005)), columns=fiber[check.values>0.006].columns)
        fiber = fiber.append(df, ignore_index=False)
        fiber = fiber.sort_index()
        
        # threshold switching issue outlier
        check = fiber.where(fiber.diff().abs()>0.1).dropna().index # wavelenth shift < 0.1nm
        fiber.iloc[check,:] = np.nan

        return fiber
    
    # calculate the Euclidean distance between two vectors
    def euclidean_distance(row1, row2):
    	distance = 0.0
    	for i in range(len(row1)-1):
    		distance += (row1[i] - row2[i])**2
    	return np.sqrt(distance) 
    
    # Locate the most similar neighbors
    def get_neighbors(train, test_row, num_neighbors):
        distances = list()
        for idx, train_row in enumerate(train):
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist, idx))
            distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        neighbors_idx = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
            neighbors_idx.append(distances[i][-1])
        return neighbors, neighbors_idx
    
    def FWHM(arr_x, arr_y):
        difference = max(arr_y) - min(arr_y)
        HM = difference / 2
        
        pos_extremum = arr_y.argmax() 
        if pos_extremum == 0:
            return np.nan
        else:
            nearest_above = (np.abs(arr_y[pos_extremum:-1] - HM)).argmin()
            nearest_below = (np.abs(arr_y[0:pos_extremum] - HM)).argmin()
            
            FWHM = (np.mean(arr_x[nearest_above + pos_extremum]) - 
                    np.mean(arr_x[nearest_below]))
            return FWHM
        
    def load_from_npz(filename):
        data = np.load(filename, allow_pickle=True)
        df = pd.DataFrame(data=data['wav'], 
                              columns=[f'sensor{i + 1}' for i in range(25)], 
                              index=data['timestamp'])
        return df
    
    def normalize_dataset(data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (data_table[col].max() - data_table[col].min())
        return dt_norm
    
    # Calculate the distance between rows.
    def distance(rows, d_function='euclidean'):
        if d_function == 'euclidean':
            # Assumes m rows and n columns (attributes), returns and array where each row represents
            # the distances to the other rows (except the own row).
            return scipy.spatial.distance.pdist(rows, 'euclidean')
        else:
            raise ValueError("Unknown distance value '" + d_function + "'")
            
    def save_events(count, timestamp, wav1, wav2, fiber1_id, fiber2_id, fiber1_sensors, fiber2_sensors):
    # save event to npz file
        event_filename = 'evt_'+timestamp+'_L23_E'+'%04d'%count+'.npz'
        np.savez_compressed(event_filename,
                            timestamp=timestamp,
                            wav1= wav1,
                            wav2= wav2,
                            event_id=count,
                            fiber1_id=fiber1_id,
                            fiber2_id=fiber2_id,
                            fiber1_sensors=fiber1_sensors,
                            fiber2_sensors=fiber2_sensors)


    
class UnionFind:

    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.
    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def add(self, object, weight):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = weight

    def __contains__(self, object):
        return object in self.parents

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            assert(False)
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure.

        """
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.parents[r] = heaviest