import networkx as nx
import pandas as pd
import itertools
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# this script is copied from:
# https://stackoverflow.com/questions/50634876/how-can-you-remove-superset-lists-from-a-list-of-lists-in-python
def get_minimal_subsets(sets):
    """"
    Parameters
    ----------
    sets: list
    This function removes all subsets from a list of tuples."""
    sets = sorted(map(set, sets), key=len)
    minimal_subsets = []
    for s in sets:
        if not any(minimal_subset.issubset(s) for minimal_subset in minimal_subsets):
            minimal_subsets.append(s)
    # added, converge to tuple
    tuples_inside = [tuple (k) for k in minimal_subsets]

    return tuples_inside


def random_vessel_generator(df_prob, load):
    """ Function to generate random vessel data.
    Parameters
    ----------
    df_prob: pd.DataFrame
    This dataframe must contain historical travel data
    load: float
    This float is equal to the simulated occupation of the network and may be read as the part of the traffic that is
    electric.
    """
    # create dict to store random prob based values later on
    main_dict = {i: [] for i in df_prob.columns}

    # copy origin and destination from original df
    main_dict['origin'] = list(df_prob.origin)
    main_dict['destination'] = list(df_prob.destination)
    main_dict['route_v'] = list(df_prob.route_v)
    main_dict['trip_count'] = []

    # fill nan values and create type list
    df_prob.fillna(0, inplace=True)
    type_list = list(df_prob.iloc[:, 3:-1])
    total = df_prob.trip_count.sum()

    # determine parameters randomly drawing ODs
    prob_r = [i / total for i in df_prob.trip_count]

    to_pick_r = list(np.arange(0, len(df_prob)))
    count_r = total * load

    # randomly draw ODs
    rand_routes = np.random.choice(a=to_pick_r, size=round(count_r), replace=True, p=prob_r)
    unique_r, counts_r = np.unique(rand_routes, return_counts=True)
    temp_type_dict_r = dict(zip(unique_r, counts_r))

    # for each OD pair
    for pair in range(len(df_prob)):

        # store trip count data previous draw
        if pair in temp_type_dict_r.keys():
            main_dict['trip_count'].append(temp_type_dict_r[pair])
        # chance that nothing is drawn, then append 0
        else:
            main_dict['trip_count'].append(0)

        # determine parameters to randomly draw ship types for each OD pair
        prob = list(df_prob.iloc[pair, 3:-1].values/df_prob.iloc[pair, 3:-1].sum())
        to_pick = type_list
        count = main_dict['trip_count'][pair]

        # randomly draw ship types for each OD pair
        rand_vessels = np.random.choice(a=to_pick, size=round(count), replace=True, p=prob)
        unique, counts = np.unique(rand_vessels, return_counts=True)
        temp_type_dict = dict(zip(unique, counts))

        # append amount of random generated vessels right dict list
        for key in type_list:
            if key in temp_type_dict.keys():
                main_dict[key].append(temp_type_dict[key])
            else:
                main_dict[key].append(0)

    # now make dict
    df_return = pd.DataFrame.from_dict(main_dict)

    return df_return


def flow_computation(df):
    """
    Parameters
    ----------
    df: pd.Dataframe
    This dataframe is compiled using the random_vessel_generator."""

    ship_data = {'M0': 0.0,
 'M1': 1.0,
 'M2': 1.7142857142857142,
 'M3': 2.4857142857142858,
 'M4': 2.4857142857142858,
 'M5': 2.4857142857142858,
 'M6': 3.942857142857143,
 'M7': 3.942857142857143,
 'M8': 8.142857142857142,
 'M9': 8.142857142857142,
 'M10': 11.514285714285714,
 'M11': 11.514285714285714,
 'M12': 11.514285714285714,
 'C1b': 1.0,
 'C1l': 1.0,
 'C2l': 3.942857142857143,
 'C3l': 0.0,
 'C2b': 11.514285714285714,
 'C3b': 11.514285714285714,
 'C4': 0.0,
 'B01': 1.0,
 'B02': 1.4285714285714286,
 'B03': 0.0,
 'B04': 2.4857142857142858,
 'BI': 3.942857142857143,
 'BII-1': 8.142857142857142,
 'BIIa-1': 8.142857142857142,
 'BIIL-1': 8.142857142857142,
 'BII-2L': 0.0,
 'BII-2b': 11.514285714285714,
 'BII-4': 0.0,
 'BII-6b': 0.0,
 'BII-6l': 0.0}
    # ship_data = pd.read_excel('data/ship_types.xlsx')
    # ship_data.fillna(0, inplace=True)
    # ship_data = dict(zip(ship_data['RWS-class'], ship_data['Factor']))
    # pickle.dump(ship_data, open("../data/flow_comp_factors.p", "wb"))

    # create dict to store path based values
    flows = {}
    # loop over data frame
    for i in range(len(df)):
        # subset all data ship type data
        a = df.iloc[:, 3:-1]
        # flow is initially 0
        flow = 0
        # add number of ships times specific ship type weighing factor
        for row in a.columns:
            flow += ship_data[row] * a[row][i]
        # store flow, divide by 365 to get daily flow
        flows[(df.origin[i], df.destination[i], df.route_v[i])] = (flow/365)
    return flows


def first_stage_frlm(r, G, OD, paths, path_lengths, df_h):
    """
    Returns feasible charging station combinations for transport network G for routes in OD,
    considering travel range r, assuming that charging stations can be placed on any node of G.
    Parameters
    ----------
    r : float
        range means of transport with full tank.
    G : NetworkX graph
        Must include all origins, destinations and any nodes where a refueling station may be placed.
    OD: dict
        This dict contains the travel data within network G, travel data from A-B and from B-A should be summed up and
        entered as either one of them.
        example input:
        {(node_1, node_2, route_v1) : flow12_r1, (node_1, node_2, route_v2) : flow12_r2, (node_1, node_3, route_v1) :
        flow13_v1}
    paths: dict
        Dictionary that contains all paths between the OD pairs that are in OD.
        example input:
        {(node_1, node_2, route_v1) : [list of nodes consecutive], (node_1, node_2, route_v2) :
         [list of nodes consecutive], (node_1, node_3, route_v1) : [list of nodes consecutive]}
    path_lengths: dict
        Dictionary that contains all path lengths (in meters) between the OD pairs that are in OD, with the same keys as
        OD and paths dicts.
    df_h: pd.DataFrame
        This is a Dataframe as generated in revised_network_cleaning.ipynb, that contains the data of harbours and the
        corresponding harbour nodes in G.
        """
    # load in harbour exits that are created in notebook harbour exits
    harbour_nodes = list(df_h.harbour_node.unique())
    harbour_dict = {}
    # collect paths to refuel and path lengths in dicts, first create empty dicts

    # dict to collect eq and fq values
    dict_eq_fq = {'q': [], 'e_q': [], 'f_q': []}

    # create list with all nodes in keys and combinations
    # nodes_in_comb =[]

    # for each route
    for (origin, destination, version), flow in OD.items():
        # create empty list to store harbours on route
        harbour_dict[(origin, destination, version)] = []
        # check all nodes for harbours and append if necessary
        for node in paths[(origin, destination, version)]:
            if node in harbour_nodes:
                harbour_dict[(origin, destination, version)].append(node)

        # store relevant variables
        dict_eq_fq['q'].append((origin, destination, version))
        dict_eq_fq['f_q'].append(flow)
        dict_eq_fq['e_q'].append((1 / (max(1, int(r / (path_lengths[(origin, destination, version)] * 2))))))

    # make master dict with key q, with list of all feasible station combinations on q with r
    route_refuel_comb = {}

    for route_key, potential_locations in harbour_dict.items():
        h = []
    # functioning shortcut, check if any single station is enough (e.g. dist < 2 x r)
        # create all possible station combinations on this path
        for L in range(0, len(potential_locations) + 1):
            for k in itertools.combinations(potential_locations, (L + 1)):
                h.append(k)
        # now add to dict:
        route_refuel_comb[route_key] = h

    # now check feasibility, new master dict to store feasible combinations
    feasible_combinations = {}

    for route_key, route in route_refuel_comb.items():
        feasible_combinations[route_key] = []
        # store path for this route (on which round trip should be feasible)
        harbours_on_route = harbour_dict[route_key]
        # this creates a list with (a, b, c, b, a) if route from a to c via b.
        round_trip = paths[route_key][:-1] + paths[route_key][::-1]

        # now loop through all possible station combinations
        for combi in route:
            # print('evaluate combi', combi)
            # start at origin
            current_pos = round_trip[0]
            # start with full range if refueling station at origin, otherwise half full
            if current_pos in combi:
                current_range = r
            else:
                current_range = r * 0.5
            # simulate power levels during round trip
            # [1:] because first dest is second entry round trip list
            for sub_dest in round_trip[1:]:
                # print('currently at', current_pos, 'traveling to', sub_dest, 'current range =', current_range)
                # try to travel to new dest, first calculate dist to new destination
                dist = nx.dijkstra_path_length(G, current_pos, sub_dest, weight='length_m')
                # only travel if dist is not too long
                if (current_range - dist) >= 0:

                    # update range and pos
                    current_pos = sub_dest
                    current_range -= dist
                    # if there is a refueling station, refuel
                    if sub_dest in combi:
                        current_range = r

                    # final dest reached? (e.g. dest if refuel station at dest, otherwise origin)
                    if (current_pos in combi) and (current_pos == harbours_on_route[-1]):
                        feasible_combinations[route_key].append(combi)
                        break
                    # else: maybe feasible, double back route to check!
                    elif current_pos == harbours_on_route[0]:
                        feasible_combinations[route_key].append(combi)
                        break
                else:
                    break

    # next: find and remove supersets
    for route_key, combinations in feasible_combinations.items():
        if len(combinations) > 1:
            feasible_combinations[route_key] = get_minimal_subsets(feasible_combinations[route_key])

    # Reformat data: create two dicts one with b_qh values and one with g_qhk values
    # first create list of all possible combinations
    unique_combinations = []
    for i in feasible_combinations.values():
        unique_combinations += i
    # remove duplicates
    unique_combinations = list(set(unique_combinations))

    # setup empty dict with keys to fill with b_qh values
    dict_b = {'q': []}
    # column for each unique combi
    for combi in unique_combinations:
        dict_b[combi] = []

    # first dict_
    for route_key, combinations_route in feasible_combinations.items():
        dict_b['q'].append(route_key)
        for combination in unique_combinations:
            if combination in combinations_route:
                dict_b[combination].append(1)
            else:
                dict_b[combination].append(0)

    # setup next dict to store g_qhk values
    dict_g = {'q': [], 'h': []}
    for node in harbour_nodes:
        dict_g[node] = []
    # fill second dict for g_qhk
    # print('Combinations:', combinations)
    for route_key, combinations in feasible_combinations.items():
        for combination in combinations:
            dict_g['q'].append(route_key)
            dict_g['h'].append(combination)
            for node in harbour_nodes:
                # create keys on run for now
                # if not node in dict_g.keys():
                #     dict_g[node] = []
                # print('Node:', node, 'combination:', combination)
                if node in combination:
                    if (node == harbour_dict[route_key][0]) or (node == harbour_dict[route_key][-1]):
                        dict_g[node].append(1)
                    else:
                        dict_g[node].append(2)
                else:
                    dict_g[node].append(0)

    # create dicts to return and set index right
    df_b = pd.DataFrame.from_dict(dict_b)
    df_b.set_index('q', inplace=True)

    df_g = pd.DataFrame.from_dict(dict_g)
    df_g.set_index(['q', 'h'], inplace=True)

    df_eq_fq = pd.DataFrame.from_dict(dict_eq_fq)
    df_eq_fq.set_index('q', inplace=True)

    return df_b, df_g, df_eq_fq


from pulp import *
import re


def second_stage_frlm(p, c, max_per_loc, df_g, df_b, df_eq_fq):
    """ This program optimally sites n charging stations with a max capacity c,
    based on three DataFrames that are generated by the first_stage_FRLM function.
        Parameters
        ----------
        p : int
            #charging stations modules to locate on any node of G.

        c : float
            max (average) flow a charging station can cope with.
        max_per_loc: int
        Maximum number of charging modules that can be placed at a certain location.

        ## the following three inputs are data frames that can be generated using the first_stage_FRLM function
        df_g : pd.DataFrame()
            DataFrame with a row for each route q and a column for each existing charging station combination h.
            b_qh = 1 combination h can support trips on path h, 0 otherwise.

        df_b : pd.DataFrame()
            DataFrame with a row for each charging station combination h, and a column for each unique facility k.
            a_qh = 1 if combination k is in combination h, 0 otherwise.

        df_eq_fq : pd.DataFrame()
            Dataframe with a row for each route q, that contains two columns corresponding f_q and e_q values.
        """

    # define y_qh for each q and each h, and restrict between 0 and 1
    # constraint 1.5 already incorporated
    # create list of index to be able to loop over double index
    a = df_g.reset_index()
    flow_a = []
    for i in a.index:
        flow_a.append((a.q[i], a.h[i]))

    # first decision variable: allocated flow
    flow_allocation = pulp.LpVariable.dicts("Flow_captured",
                                            ((q, h) for q, h in flow_a),
                                            lowBound=0,
                                            upBound=1,
                                            cat='Continuous')

    # second decision variable: number of facilities to place at each site is also decision var
    facilities_to_build = pulp.LpVariable.dicts("Facilities",
                                                (facility for facility in df_g.columns),
                                                lowBound=0,
                                                upBound=max_per_loc,
                                                cat='Integer')
    # problem definition
    model = LpProblem('CFRLM', LpMaximize)

    # objective function
    model += pulp.lpSum([flow_allocation[q, h] * df_b[h][q] * df_eq_fq['f_q'][q] for q, h in flow_a])

    # ###############################################constraints##################################################
    # first constraint
    # for each facility
    for key, facility in facilities_to_build.items():
        model += pulp.lpSum(df_eq_fq['e_q'][q] * df_g[key].loc[df_g.index == (q, h)] * df_eq_fq['f_q'][q] *
                            flow_allocation[q, h] for q, h in df_g.index) <= pulp.lpSum(c * facility)

    # second constraint
    model += pulp.lpSum(facilities_to_build[i] for i in facilities_to_build.keys()) <= p

    # third constraint
    for q in df_b.index:
        model += pulp.lpSum([flow_allocation[q, h] * df_b[h][q]] for h in df_g.reset_index().
                            loc[df_g.reset_index().q == q].h) <= 1

    # print(model)

    # solve
    model.solve()

    # status = LpStatus[model.status]
    # print(status)

    supported_flow = value(model.objective)

    return supported_flow


def flow_refueling_location_model(load, r, stations_to_place, station_cap, max_per_loc):
    """abc
    Parameters
    ----------
    load:float
    r:int
    stations_to_place:int
    station_cap: int
    max_per_loc: int
    paths: dict
    path_lengths: dict
    df_h: pd.DataFrame()
    df_ivs: pd.DataFrame()
    G: nx.Graph()
    """
    G = pickle.load(open('data/cleaned_network.p', 'rb'))
    df_h = pickle.load(open("data/harbour_data_100.p", "rb"))
    df_ivs = pickle.load(open("data/ivs_exploded_100.p", "rb"))
    paths = pickle.load(open("data/paths_ship_specific_routes.p", "rb"))
    path_lengths = pickle.load(open("data/path_lengths_ship_specific_routes.p", "rb"))

    df_random = random_vessel_generator(df_ivs, load)
    flows = flow_computation(df_random)
    total_flow = sum(flows.values())
    df_b, df_g, df_eq_fq = first_stage_frlm(r, G, OD=flows, paths=paths, path_lengths=path_lengths, df_h=df_h)
    supported_flow = second_stage_frlm(stations_to_place, station_cap, max_per_loc, df_g, df_b, df_eq_fq)
    supported_fraction = (supported_flow/total_flow)

    return total_flow, supported_fraction

