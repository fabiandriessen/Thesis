from multiprocessing import freeze_support
from mesa.batchrunner import batch_run
from model_new_v import VesselElectrification
import pickle
import numpy as np
from create_input_data_abm import create_input_data_abm
import pandas as pd


def create_key(o, d, r_v):
    key1 = (o, d, r_v)
    return key1


pick_from = np.linspace(0, 1000000, 1000001)
seeds = np.random.choice(a=pick_from, size=100, replace=False)
seeds = list(seeds)
seeds = [round(i) for i in seeds]

# import dataframe with input parameters for each run
df_9scenarios = pd.DataFrame(pickle.load(open("data/data_9_scenarios.p", "rb")))
G = pickle.load(open('data/network.p', 'rb'))
paths = pickle.load(open('data/paths.p', 'rb'))
df_random = pickle.load(open('data/df_random.p', 'rb'))


for i, row in df_9scenarios.iterrows():

    df_abm = create_input_data_abm(G, paths, row['non_zero_flows'], row['optimal_facilities'])
    # configure df random for abm
    df_random['key'] = df_random.apply(lambda x: create_key(x.origin, x.destination, x.route_v), axis=1)
    df_random = df_random.loc[df_random.key.isin(row['non_zero_flows'].keys())]
    df_random = df_random.loc[df_random.trip_count != 0]

    pickle.dump(df_random, open('data/df_random_batch.p', 'wb'))
    df_abm.to_csv('data/df_abm_batch.csv')
    pickle.dump(row['non_zero_flows'], open('data/non_zero_flows_batch.p', 'wb'))

    params = {'c': row['c'], 'r': row['r'], 'seed': seeds}

    if __name__ == '__main__':
        freeze_support()
        result = batch_run(VesselElectrification,
                           iterations=1,
                           parameters=params,
                           data_collection_period=(60 * 24 * 2),
                           max_steps=(60 * 24 * 2),
                           number_processes=20,
                           display_progress=True)

        pickle.dump(result, open('data/batch_9scenarios'+str(row['r'])+str(row['c'])+str(row['m'])+'.p', 'wb'))
