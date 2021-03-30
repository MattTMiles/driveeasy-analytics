from datetime import timedelta, datetime, timezone
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pydea.ETL.wav import *

'''
Takes in a series of npz files and returns an output log file of:

 - Location (Which Road)
 - Date (YYYY:MM:DD)
 - Time (HH:MM:SS.mmm)
 - Lane (Lane detection package)
 - Error Code (Needs clarification)
 - Direction (Event Detection?)
 - Speed (Speed Algorithms)
 - Position (Position in the lane (May need clarification))
 - Width (Unsure if exists)
 - Class (Class detection)
 - Axles (Axle Detection)
 - Axle Groups (How many axle groups)
 - d1 (Distance between the first and second axle group)
 - d2 (Distance between the second and third axle group (If applicable))
 - Length (Vehicle Length)
'''

#At the moment the location will have to be manually entered as the 
# *npz files have no identity info

def output(data_dir, location, minutedelta):

    wav_files = list(data_dir.glob('*.npz'))
    
    df = load_wav_into_dataframe(wav_files[0])

    initial = df.index[0]
    finish = initial + timedelta(minutes = minutedelta)

    while initial < df.index[-1]:
        output_data = []
        output_data = df[df.index >= initial]
        output_data = output_data[output_data.index < finish]

        to_csv = []
        
        date = initial.date()
        date = date.isoformat()

        time = initial.time()
        time = time.isoformat()

        
        
        to_csv = [location, date, time, lane, err, direction, speed, position, width, v_class, num_ax, num_ax_gr, d1, d2, Length]
        labels = ['Location','Date','Time','Lane','Error Code','Direction','Speed','Position','Width','Class','Axles','Axle Groups','d1','d2','Length']

        pd_df = pd.DataFrame(to_csv,columns=labels)
        pd_df.to_csv(str(initial))

        initial = finish
        finish = initial + timedelta(minutes = minutedelta)




    newdf = []

    newdf['Location'] = loc
    newdf = 
