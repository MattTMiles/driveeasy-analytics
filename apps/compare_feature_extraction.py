import streamlit as st
import pandas as pd
import numpy as np
import scipy.signal as signal
import datetime
from pathlib import Path
from sklearn.preprocessing import minmax_scale

import plotly.express as px
from pydea.viz import plot_heatmap
from pydea.feature.speed import SpeedEstimationHY, SpeedEstimationQC, SpeedEstimationJY

