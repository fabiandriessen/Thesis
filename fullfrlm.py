from flow_computation import flow_computation
from random_vessel_generator import random_vessel_generator
from first_stage_frlm import first_stage_frlm
from second_stage_frlm import second_stage_frlm
from generate_network import generate_network
from visualize_placement import visualize_placement
import pickle


def flow_refueling_location_model(load, r, stations_to_place, station_cap, max_per_loc, additional_nodes=0,
                                  vis=False):
    """abc
    Parameters
    ----------
    load:float
        Percentage of vessels on the network compared to the 2021 total.
    r:int
        Range of a vessel.
    stations_to_place:int
        Total number of charging modules to place.
    station_cap: int
        Maximum capacity of a charging station per time unit.
    max_per_loc: int
        Maximum number of charging modules that may be placed at a location.
    additional_nodes: int
        Number of additional nodes that should be inserted into the original network.
    vis: Boolean
    If this variable is True, a visualisation is presented
    """
    G = pickle.load(open('data/network_cleaned_final.p', 'rb'))
    df_h = pickle.load(open("data/revised_cleaning_results/harbour_data_100.p", "rb"))
    df_ivs = pickle.load(open("data/revised_cleaning_results/ivs_exploded_100.p", "rb"))
    path_lengths = pickle.load(open("data/revised_cleaning_results/path_lengths_ship_specific_routes.p", "rb"))
    paths = pickle.load(open('data/final_paths.p', "rb"))

    # if additional nodes need to be considered, update G, paths and inserted accordingly
    inserted = []
    if additional_nodes != 0:
        G, paths, inserted = generate_network(G, paths, additional_nodes)

    # generate random data
    df_random = random_vessel_generator(df_ivs, load)
    flows = flow_computation(df_random)
    total_flow = sum(flows.values())

    # execute first stage, with or without additional nodes
    df_b, df_g, df_eq_fq = first_stage_frlm(r, G, OD=flows, paths=paths, path_lengths=path_lengths, df_h=df_h,
                                            additional_nodes=inserted)

    # execute second stage
    optimal_facilities, optimal_flows, non_zero_flows, supported_flow, routes_supported = second_stage_frlm(stations_to_place,
                                                                                          station_cap, max_per_loc,
                                                                                          df_g, df_b, df_eq_fq)
    supported_fraction = (supported_flow/total_flow)

    if vis:
        visualize_placement(G, flows, optimal_facilities, non_zero_flows, df_h, paths, unused=True)

    return total_flow, supported_fraction, optimal_facilities, non_zero_flows, routes_supported

