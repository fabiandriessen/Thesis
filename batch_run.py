import pandas as pd
from first_stage_frlm import first_stage_frlm
from second_stage_frlm import second_stage_frlm
from flow_computation import flow_computation
from random_vessel_generator import random_vessel_generator
import pickle

G = pickle.load(open('data/revised_cleaning_results/cleaned_network.p', 'rb'))
df_h = pickle.load(open("data/revised_cleaning_results/harbour_data_100.p", "rb"))
df_ivs = pickle.load(open("data/revised_cleaning_results/ivs_exploded_100.p", "rb"))
paths = pickle.load(open("data/revised_cleaning_results/paths_ship_specific_routes.p", "rb"))
path_lengths = pickle.load(open("data/revised_cleaning_results/path_lengths_ship_specific_routes.p", "rb"))


def batch_run(df_p, loads=None,ranges=None, iterations = None):

    # lists to store values to plot
    if ranges is None:
        ranges = [50000, 100000, 150000, 200000, 250000]
    if loads is None:
        loads = [0.1, 0.3, 1]
    if iterations is None:
        iterations = 2

    plot_dict = {'load': [], 'max_modules': [], 'v_range': [], 'n_stations': [], 'station_capacity': [],
                 'total_supported_flow': [], 'iteration': []}
    for iteration in range(iterations):
        for load in loads:
            df_random1 = random_vessel_generator(df_p, load)
            flows = flow_computation(df_random1)
            for v_range in ranges:
                # generate random df and store get path pased flows
                df_b, df_g, df_eq_fq = first_stage_frlm(v_range, G, OD=flows, paths=paths, path_lengths=path_lengths,
                                                        df_h=df_h)
                for max_modules in [10, 20, 40, 60, 80, 100, 200, 300, 350]:
                    for n_stations in [1, 5, 10]:
                        # store placed stations
                        # run first stage FRLM
                        for station_cap in [1, 5, 10]:
                            # run second stage FRLM
                            optimal_facilities, optimal_flows, non_zero_flows, supported_flow = second_stage_frlm(
                                n_stations, station_cap, max_modules, df_g, df_b, df_eq_fq)
                            # store relevant values
                            plot_dict['load'].append(load)
                            plot_dict['max_modules'].append(max_modules)
                            plot_dict['v_range'].append(v_range)
                            plot_dict['n_stations'].append(n_stations)
                            plot_dict['station_capacity'].append(station_cap)
                            plot_dict['total_supported_flow'].append(supported_flow)
                            plot_dict['iteration'].append(iteration)

    df_batch = pd.DataFrame.from_dict(plot_dict)

    return df_batch

