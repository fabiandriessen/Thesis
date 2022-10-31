import networkx as nx
import pandas as pd
import itertools
import pickle


# this script is copied from:
# https://stackoverflow.com/questions/50634876/how-can-you-remove-superset-lists-from-a-list-of-lists-in-python
def get_minimal_subsets(sets):
    """
    This function removes all subsets from a list of tuples.
    Parameters
    ----------
    sets: list
    """
    sets = sorted(map(set, sets), key=len)
    minimal_subsets = []
    for s in sets:
        if not any(minimal_subset.issubset(s) for minimal_subset in minimal_subsets):
            minimal_subsets.append(s)
    # added, converge to tuple
    tuples_inside = [tuple(k) for k in minimal_subsets]

    return tuples_inside


def first_stage_frlm(r, G, OD, paths, path_lengths, df_h, additional_nodes=None):
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

    additional_nodes: list
    additional_nodes: list
        This is a list that should contain all additional harbour nodes to be considered, next to the origin and
        destination harbours.
        """

    # load in harbour exits that are created in notebook harbour exits
    if additional_nodes is None:
        additional_nodes = []

    harbour_nodes = list(df_h.harbour_node.unique()) + additional_nodes
    harbour_dict = {}
    # collect paths to refuel and path lengths in dicts, first create empty dicts

    # dict to collect eq and fq values
    dict_eq_fq = {'q': [], 'e_q': [], 'f_q': []}

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

        # adjusted compared to original: no roundtrip assumed,single  path length used and no integer value used
        dict_eq_fq['e_q'].append(1 / (max(1, int(r / (path_lengths[(origin, destination, version)])))))

    # make master dict with key q, with list of all feasible station combinations on q with r
    route_refuel_comb = {}

    for route_key, potential_locations in harbour_dict.items():
        h = []
        # create all possible station combinations on this path
        for L in range(0, min(4, (len(potential_locations) + 1))):
            for k in itertools.combinations(potential_locations, (L + 1)):
                h.append(k)
        # now add to dict:
        route_refuel_comb[route_key] = h

    # now check feasibility, new master dict to store feasible combinations
    feasible_combinations = {}

    for route_key, route in route_refuel_comb.items():
        # print('Evaluate route', route_key, route)
        feasible_combinations[route_key] = []
        # store path for this route (on which round trip should be feasible)
        harbours_on_route = harbour_dict[route_key]
        # print('Harbours on route:', harbours_on_route)
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
                # try to travel to new dest, first calculate dist to new destination
                dist = nx.dijkstra_path_length(G, current_pos, sub_dest, weight='length_m')
                # print('currently at', current_pos, 'traveling to', sub_dest, 'current range =', current_range,
                # 'distance', dist)

                # only travel if dist is not too long
                if (current_range - dist) >= 0:

                    # update range and pos
                    current_pos = sub_dest
                    current_range -= dist
                    # if there is a refueling station, refuel
                    if sub_dest in combi:
                        current_range = r

                    # final dest reached? (e.g. dest if refuel station at dest, otherwise origin)
                    if (current_pos in combi) and (current_pos == paths[route_key][-1]):
                        feasible_combinations[route_key].append(combi)
                        break
                    # else: maybe feasible, double back route to check!
                    elif current_pos == paths[route_key][0]:
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
                    if (node == paths[route_key][0]) or (node == paths[route_key][-1]):
                        dict_g[node].append(1)
                    else:
                        # changed compared to original CFRLM: no round trip assumed thus 1 instead of 2
                        dict_g[node].append(1)
                else:
                    dict_g[node].append(0)

    # create dicts to return and set index right
    df_b = pd.DataFrame.from_dict(dict_b)
    df_b.set_index('q', inplace=True)

    df_g = pd.DataFrame.from_dict(dict_g)
    df_g.set_index(['q', 'h'], inplace=True)

    df_eq_fq = pd.DataFrame.from_dict(dict_eq_fq)
    df_eq_fq.set_index('q', inplace=True)

    return df_b, df_g, df_eq_fq, feasible_combinations