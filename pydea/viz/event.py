
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List,Sequence,Dict,Union
from pydea.datamodels.datamodel import Event

def plot_events(events:List[Event], wls_df=None):
    fig = px.line(wls_df)
    for event in events:
        start = event.timestamp[0]
        end = event.timestamp[-1]
        fig.add_vline(x=start)
        fig.add_shape(type="line",
                      xref="x", yref="y",
                      x0=start, y0=0, x1=start, y1=1,
                      line=dict(
                          color="LightSeaGreen",
                          width=3,
                      ))
        fig.add_shape(type="line",
                      xref="x", yref="y",
                      x0=end, y0=0, x1=end, y1=2,
                      line=dict(
                          color="Blue",
                          width=3,
                      ))
    plot(fig,filename='events_and_wls.html')

