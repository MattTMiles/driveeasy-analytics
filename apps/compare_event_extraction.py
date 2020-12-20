import streamlit as st
from pathlib import Path
import plotly.express as px
from pydea.viz import plot_heatmap
from pydea.ETL.wav import  load_wav_into_dataframe
from pydea import preprocessing as pp
from pydea.preprocessing.detrend import detrend_df
from pydea.preprocessing import remove_outliers
import pydea.preprocessing.remove_outliers as rmout

# data_file = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\pydea\datasets\francis\wav\wav_20201107_165626_F1.npz')
data_file = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\data\M80\wav_20201124_195912_F03_UTC.npz')
import numpy

def asStride(arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides=(stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs=numpy.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs

def poolingOverlap(mat,ksize,stride=None,method='max',pad=False):
    '''Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize
    if stride is None:
        stride=(ky,kx)
    sy,sx=stride

    _ceil=lambda x,y: int(numpy.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,sy)
        nx=_ceil(n,sx)
        size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad=numpy.full(size,numpy.nan)
        mat_pad[:m,:n,...]=mat
    else:
        mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view=asStride(mat_pad,ksize,stride)

    if method=='max':
        result=numpy.nanmax(view,axis=(2,3))
    elif method=='min':
        result = numpy.nanmin(view, axis=(2,3))
    else:
        result=numpy.nanmean(view,axis=(2,3))

    return result
def pooling(mat,ksize,method='max',pad=False):
    '''Non-overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           n//f, n being <mat> size, f being kernel size.
           if pad, output has size ceil(n/f).

    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(numpy.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=numpy.full(size,numpy.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    if method=='max':
        result=numpy.nanmax(mat_pad.reshape(new_shape),axis=(1,3))
    else:
        result=numpy.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

@st.cache
def load_data(data_file, load_quantile=0.2):
    df = load_wav_into_dataframe(data_file)
    wls = detrend_df(df) * 1000
    wls = remove_outliers_df(wls, 1000)

    return  wls

def remove_outliers_df(df, outlier_threshold=1000):
    df[df > outlier_threshold] = 0
    df[df < -outlier_threshold] = 0
    # print(df)
    return df


def line_plot(min_agg):
    fig = px.line(min_agg)
    return fig


wls0 = load_data(data_file)
wls = wls0.iloc[:,0:11]

st.write(wls.head())
import pandas as pd
wls_pool = pooling(wls.values,ksize=[25,10],method='min')
st.write(wls_pool[0:5])
fig = line_plot(wls_pool)
st.write(fig)

ts = numpy.arange(wls_pool.shape[0])
ts = wls.index[0:wls_pool.shape[0]]
wls_pool_df = pd.DataFrame(index=ts, data=wls_pool)
fig = plot_heatmap(wls_pool_df)
st.write(fig)



min_agg = wls.min(axis=1)




fig1 = line_plot(min_agg)
st.write(fig1)
# fig = plot_heatmap(wls)
# st.write(fig)