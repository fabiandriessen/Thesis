from flow_computation import flow_computation
from random_vessel_generator import random_vessel_generator
from first_stage_frlm import first_stage_frlm
from second_stage_frlm import second_stage_frlm
from generate_network_nodes import generate_network
from visualize_placement import visualize_placement
from create_input_data_abm import create_input_data_abm
from additional_intersections_new import additional_intersections
import pickle


def create_key(o, d, r_v):
    key1 = (o, d, r_v)
    return key1


def flow_refueling_location_model(r, p, scenario_var, additional_nodes=0, n=0, vis=False, o=24,
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

    n: int
        Number of additional intersections to include

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
    if scenario_var == 0:
        c = 2000
        x_m = 5
    elif scenario_var == 1:
        c = 3333
        x_m = 3
    elif scenario_var == 2:
        c = 10000
        x_m = 1
        # no additional nodes necessary if higher than

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

    if additional_nodes == 3:
        G, paths, inserted = generate_network(G, paths, n)
        inserted += ['8863114', '8864435', '8865887', '8861447', '8864526', '8860631', '8863272', '8863762', '8861611',
                     '8860623', '8864952', '8864915', '8860563', '8860918', '8862907', '8865462', '8860579', '8861863',
                     '8866661', '8864151', '8864912', '8864258', '8861499', '8864553', '8865119']
    elif additional_nodes == 2:
        G, paths, inserted = generate_network(G, paths, n)
    elif additional_nodes == 1:
        inserted += ['8863114', '8864435', '8865887', '8861447', '8864526', '8860631', '8863272', '8863762', '8861611',
                     '8860623', '8864952', '8864915', '8860563', '8860918', '8862907', '8865462', '8860579', '8861863',
                     '8866661', '8864151', '8864912', '8864258', '8861499', '8864553', '8865119']

    # execute first stage, with or without additional nodes
    df_b, df_g, df_eq_fq, feasible_combinations = first_stage_frlm(r, G, OD=flows, paths=paths,
                                                                   path_lengths=path_lengths, df_h=df_h,
                                                                   additional_nodes=inserted)

    # execute second stage
    optimal_facilities, optimal_flows, non_zero_flows, supported_flow, routes_supported = second_stage_frlm(
        p, x_m, c, o, df_g, df_b, df_eq_fq)

    # collect data
    total_flow = sum(flows.values())

    max_supported = {i: flows[i] for i in flows if len(feasible_combinations[i]) > 0}
    max_supported = sum(max_supported.values())

    fraction_captured_total = (supported_flow / total_flow)

    serviceable_fraction = (max_supported / total_flow)

    served_fraction = (supported_flow / max_supported)

    used_facilities = [i for i, j in optimal_facilities.items() if j > 0]
    add_nodes_used = float(len(list(set(used_facilities) - set(df_h.harbour_node.to_list()))))

    if vis:
        visualize_placement(G, flows, optimal_facilities, non_zero_flows, df_h, paths, unused=True)

    return total_flow, fraction_captured_total, serviceable_fraction, served_fraction, optimal_facilities, \
        non_zero_flows, add_nodes_used
