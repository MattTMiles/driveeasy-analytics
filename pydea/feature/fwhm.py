import numpy as np

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
