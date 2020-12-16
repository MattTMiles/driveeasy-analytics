from pydea.datamodels.datamodel import Raw
from pathlib import Path
from pydea.viz.figureX import create_browse_figure
import matplotlib as mpl
mpl.use('TkAgg')

raw_file = Path(r'C:\Users\hyu\gitlab_repos\driveeasy-analytics\data\M80\333Hz\raw_2020-12-12_19-00-24.gzip')
raw = Raw(raw_file)
params = dict(inst=raw)
fig = create_browse_figure(**params)
