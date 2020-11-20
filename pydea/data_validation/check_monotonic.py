import pandas as pd
import numpy as np

def check_monotonic_increase(arr):
    dx = np.diff(arr)
    return np.all(dx>=0)

def check_monotonic_increase(arr):
    dx = np.diff(arr)
    return np.all(dx>=0)