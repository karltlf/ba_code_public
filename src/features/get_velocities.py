import numpy as np
import pandas as pd



def get_velocities_in_ms(df_cuid_grouped:pd.DataFrame) -> list[np.array]:
    '''
    Returns calculated total velocity measurement in m/s from velocity in x and y direction

    :param df_cuid_grouped: pandas Dataframe grouped by track_id with 'vx' and 'vy' features
    :return velocities_calculated_m_s: List of numpy arrays of velocities corresponding to x,y coordinates
    '''
    vx = [np.array(vx) for vx in df_cuid_grouped['vx']]
    vy = [np.array(vy) for vy in df_cuid_grouped['vy']]

    velocities_calculated_m_s = [0] * len(vx)
    for i in range(len(vx)):
        velocity = np.sqrt(vx[i]**2 + vy[i]**2)
        velocities_calculated_m_s[i] = velocity

    return velocities_calculated_m_s

def get_velocities_in_kmh(df_cuid_grouped:pd.DataFrame) -> list[np.array]:
    velocities_calculated_m_s = get_velocities_in_ms(df_cuid_grouped)
    velocities_calculated_km_h = [vel*3.6 for vel in velocities_calculated_m_s]
    return velocities_calculated_km_h