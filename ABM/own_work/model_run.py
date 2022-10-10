from model_new_v import VesselElectrification
import pickle
import numpy as np
from create_input_data_abm import create_input_data_abm
import pandas as pd


"""
    Run simulation
    Print output at terminal
"""

# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick 1 minute
run_length = 8 * 24 * 60

# run time 1000 ticks
# run_length = 1000

seed = 1234567


def create_key(o, d, r_v):
    key1 = (o, d, r_v)
    return key1


pick_from = np.linspace(0, 1000000, 1000001)
seeds = np.random.choice(a=pick_from, size=17, replace=False)
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

    sim_model = VesselElectrification(input_df=df_random, df_abm=df_abm,
                                      opt_flows=row['non_zero_flows'], c=row['c'], r=row['r'], seed=seed)

    # Check if the seed is set
    print("SEED " + str(sim_model._seed))

    # One run with given steps
    for i in range(run_length):
        sim_model.step()

    agent_data = sim_model.datacollector.get_agent_vars_dataframe()
    model_data = sim_model.datacollector.get_model_vars_dataframe()

    pickle.dump(agent_data, open('data/agent_data.p', "wb"))
    pickle.dump(model_data, open('data/model_data.p', "wb"))
