import pandas as pd
import pickle


def get_vessel_data_batch(df_batch):
    df_batch = pd.DataFrame(df_batch)
    df_batch = df_batch.loc[df_batch.Step > 0]
    df_vessels = df_batch.groupby('RunId').last()
    vessel_df = pd.DataFrame(df_vessels['data_completed_trips'][0])
    for i in range(2, len(df_vessels['data_completed_trips'])):
        # Variable i is not used now!!
        df_temp = pd.DataFrame(df_vessels['data_completed_trips'][i])
        vessel_df = pd.concat([vessel_df, df_temp])
    return vessel_df


def get_cs_data_batch(df_batch):
    df_batch = pd.DataFrame(df_batch)
    df_batch = df_batch.loc[df_batch.Step>0]
    df_charging_stations = df_batch.loc[df_batch.charging_stations > 0]
    # df_charging_stations = df_charging_stations.drop(columns=['RunId', 'iteration', 'Step', 'seed'])
    return df_charging_stations

