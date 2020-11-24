import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

# sns.set_context('talk')
plt.style.use('seaborn')


def create_XYZ_data(event):
    time_index = event.index
    time_arr = (time_index - time_index[0]).total_seconds()
    location_arr = np.arange(1, event.shape[1] + 1)
    X, Y = np.meshgrid(location_arr, time_arr)
    Z = event.values
    return X, Y, Z


def plot_heatmap(event, figure_filename=None, figsize=[500,2000]):
    X, Y, Z = create_XYZ_data(event)
    Z_xr = xr.DataArray(Z,
                        dims=("time", "sensor_location",),
                        coords={"sensor_location": X[0, :], "time": Y[:, 0]})
    heatmap = px.imshow(Z_xr,
                        labels=dict(x="Sensor Location", y="Time (seconds)", color="Strain"),
                        width=figsize[0],
                        height=figsize[1],
                        aspect='auto',
                        color_continuous_midpoint=0,
                        )
    heatmap.update_xaxes(side="top")
    if figure_filename is not None:
        if not os.path.exists("heatmap_plots"):
            os.mkdir("heatmap_plots")
            print('heatmap_plots folder created.')
        heatmap.write_image(f'heatmap_plots/{figure_filename}.png')
        print('heatmap image saved.')
    return heatmap


def plot3d_events(wls_df):
    time_index = wls_df.index
    time_arr = (time_index - time_index[0]).total_seconds()
    time_int_ind = np.arange(len(time_arr))

    location_arr = np.arange(1, 13)

    X, Y = np.meshgrid(location_arr, time_arr)
    Z = wls_df.values

    lines = []
    line_marker = dict(color='#0066FF', width=2)
    for i, j, k in zip(X, Y, Z):
        lines.append(go.Scatter3d(x=i, y=j, z=k, mode='lines', line=line_marker))
    layout = go.Layout(
        title='One Vehicle Pass Event',
        scene=dict(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        ),
        showlegend=False,
    )
    fig = go.Figure(data=lines, layout=layout)
    fig.update_layout(scene=dict(
        xaxis_title='Sensor number',
        yaxis_title='Time (seconds)',
        zaxis_title='Wavelength shift (pm)'),
        width=700,
        height=700,
        # aspect='auto',
        margin=dict(r=20, b=10, l=10, t=10))

    # plot(fig)
    # another way 3d surface
    # fig = go.Figure(data=[go.Surface(z=Z, x=time_arr, y=location_arr)])
    # fig.update_layout(title='Mt Bruno Elevation', autosize=True,
    #                   width=800, height=800,
    #                   margin=dict(l=65, r=50, b=65, t=90))
    # plot(fig)
    return fig


if __name__ == '__main__':
    pass
