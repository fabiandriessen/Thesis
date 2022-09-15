from flow_computation import flow_computation
from random_vessel_generator import random_vessel_generator
from first_stage_frlm import first_stage_frlm
from second_stage_frlm import second_stage_frlm
from generate_network import generate_network
from visualize_placement import visualize_placement
from create_input_data_abm import create_input_data_abm
from determine_additional_nodes import determine_additional_nodes
import pickle


def create_key(o, d, r_v):
    key1 = (o, d, r_v)
    return key1


def flow_refueling_location_model(r, p, c, x_m,  additional_nodes=0, vis=False, o=24,
                                  random_data=False, load=1, seed=None):
    """
    Parameters
    ----------
    r : int
        Range of a vessel.

    p:int
        Total number of charging modules to place.

    c: int
        Maximum capacity of a charging station per time unit in supplied kWh.

    x_m: int
        Maximum number of charging modules that may be placed at a location.

    additional_nodes: int
        Applied heuristic, 0 is none, 1 is 1, 2 = 2 , 3 is both.

    vis: Boolean
        If this variable is True, a visualisation is presented.

    o: float[0,24]
        Operational hours during a day.

    random_data: Boolean
        If True, run the model with random sample based instead of the empirical data.

    load:float
        Percentage of vessels on the network compared to the 2021 total to create if random data is used.

    seed:int
        Random seed to use for random data generation.
    """

    G = pickle.load(open('data/network_cleaned_final.p', 'rb'))
    df_h = pickle.load(open("data/revised_cleaning_results/harbour_data_100.p", "rb"))
    df_ivs = pickle.load(open("data/revised_cleaning_results/ivs_exploded_100.p", "rb"))
    path_lengths = pickle.load(open("data/revised_cleaning_results/path_lengths_ship_specific_routes.p", "rb"))
    paths = pickle.load(open('data/final_paths.p', "rb"))

    # generate random data
    if random_data:
        df_random = random_vessel_generator(df_ivs, seed, load)
        flows = flow_computation(df_random, r, path_lengths)
    else:
        df_random = df_ivs
        flows = flow_computation(df_random, r, path_lengths)

    inserted = []
    # include intersections if True
    if additional_nodes == 3:
        G, paths, inserted = generate_network(G, paths)
        inserted += determine_additional_nodes(G, df_h)
    elif additional_nodes == 2:
        G, paths, inserted = generate_network(G, paths)
    elif additional_nodes == 1:
        inserted += determine_additional_nodes(G, df_h)

    # execute first stage, with or without additional nodes
    df_b, df_g, df_eq_fq, feasible_combinations = first_stage_frlm(r, G, OD=flows, paths=paths,
                                                                   path_lengths=path_lengths, df_h=df_h,
                                                                   additional_nodes=inserted)

    # execute second stage
    optimal_facilities, optimal_flows, non_zero_flows, supported_flow, routes_supported = second_stage_frlm(
        r, p, x_m, path_lengths, c, o, df_g, df_b, df_eq_fq)

    # collect data
    total_flow = sum(flows.values())

    max_supported = {i: flows[i] for i in flows if len(feasible_combinations[i]) > 0}
    max_supported = sum(max_supported.values())

    fraction_captured_total = (supported_flow / total_flow)

    serviceable_fraction = (max_supported / total_flow)

    served_fraction = (supported_flow / max_supported)

    df_abm = create_input_data_abm(G, paths, non_zero_flows, optimal_facilities)

    if vis:
        visualize_placement(G, flows, optimal_facilities, non_zero_flows, df_h, paths, unused=True)

    # store range and capacity per day of a station?
    df_abm['range'] = r
    df_abm['capacity'] = c * 24

    # configure df random for abm
    df_random['key'] = df_random.apply(lambda x: create_key(x.origin, x.destination, x.route_v), axis=1)
    df_random = df_random.loc[df_random.key.isin(non_zero_flows.keys())]
    df_random = df_random.loc[df_random.trip_count != 0]

    pickle.dump(feasible_combinations, open('ABM/own_work/data/feasible_comb.p', 'wb'))
    pickle.dump(G, open("ABM/own_work/data/network.p", "wb"))
    pickle.dump(paths, open("ABM/own_work/data/paths.p", "wb"))
    pickle.dump(df_abm, open("ABM/own_work/data/df_abm.p", "wb"))
    pickle.dump(df_random, open("ABM/own_work/data/df_random.p", "wb"))
    pickle.dump(non_zero_flows, open("ABM/own_work/data/non_zero_flows.p", "wb"))
    df_abm.to_csv('ABM/own_work/data/df_abm.csv')

    return total_flow, fraction_captured_total, serviceable_fraction, served_fraction, optimal_facilities, \
        non_zero_flows, routes_supported, paths, G, df_abm, df_random, max_supported
