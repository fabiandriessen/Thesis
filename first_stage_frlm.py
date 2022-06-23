import networkx as nx
import pandas as pd
import itertools
import pickle


# this script is copied from:
# https://stackoverflow.com/questions/50634876/how-can-you-remove-superset-lists-from-a-list-of-lists-in-python
def get_minimal_subsets(sets):
    sets = sorted(map(set, sets), key=len)
    minimal_subsets = []
    for s in sets:
        if not any(minimal_subset.issubset(s) for minimal_subset in minimal_subsets):
            minimal_subsets.append(s)
    #added, converge to tuple
    tuples_inside = [tuple (k) for k in minimal_subsets]

    return tuples_inside


def first_stage_frlm(G, r, OD):
    """
    Returns feasible charging station combinations for transport network G for routes in OD,
    considering travel range r, assuming that charging stations can be placed on any node of G.
    Parameters
    ----------
    G : NetworkX graph
        must include all origins, destinations and any nodes where a refueling station may be placed.
    r : float
        range means of transport with full tank.
    OD: list
        list of travel data within network G, travel data from A-B and from B-A should be summed up and
        entered as either one of them.
        example input:
        [(node_1, node_2, flow_12),(node_1, node_3, flow_13),(node_2, node_3, flow_23)]
        """
    # load in harbour exits that are created in notebook harbour exits
    harbour_exits = pickle.load( open("data/harbour_exits.p", "rb") )

    # Now also create weighted edge list in the format [(begin, start, weight),(..), etc.]
    # for origin, destination in edge_list:
    #     edge_list_w.append((origin, destination, G.get_edge_data(origin, destination)['length_m']))

    # collect paths to refuel and path lengths in dicts, first create empty dicts
    paths = {}
    path_lengths = {}
    harbours ={}

    # dict to collect eq and fq values
    dict_eq_fq = {'q': [], 'e_q': [], 'f_q': []}

    # create list with all nodes in keys and combinations
    # nodes_in_comb =[]

    # generate the shortest paths for al origin destinations
    for origin, destination, flow in OD:
        # nodes_in_comb += [origin, destination]
        paths[(origin, destination)] = nx.dijkstra_path(G, origin, destination, weight='length_m')
        # print('All nodes on route', paths[(origin, destination)])
        # paths[(origin, destination)] = list(set(paths[(origin, destination)]).intersection(harbour_exits))

        # Put harbours on each route in list
        harbours[(origin, destination)] = []
        for node in paths[(origin, destination)]:
            if node in harbour_exits:
                harbours[(origin, destination)].append(node)

        # store path length for shortcut later on and to be able to check with range
        path_lengths[(origin, destination)] = nx.dijkstra_path_length(G, origin, destination, weight='length_m')

        # fill dict for eq_fq dataframe
        dict_eq_fq['q'].append((origin, destination))
        dict_eq_fq['f_q'].append(flow)
        dict_eq_fq['e_q'].append((1 / (max(1, int(r / (path_lengths[(origin, destination)] * 2))))))

        # Too many harbours... Q: What is the distance between harbours on each route?
        # print('Observing route:', (origin,destination))
        cleaned_harbours = [harbours[(origin, destination)][0]]
        for index in range(len(harbours[(origin, destination)])-1):
            dist = nx.dijkstra_path_length(G, cleaned_harbours[-1], harbours[(origin,destination)][index+1], weight='length_m')
            if dist > 2000:
                cleaned_harbours.append(harbours[(origin, destination)][index+1])
            # else:
                # print('node thrown out:', harbours[(origin, destination)][index+1], 'for route', origin, destination)
            # print(dist)
        harbours[(origin, destination)] = cleaned_harbours

    # print(harbours)

    # make master dict with key q, with list of all feasible station combinations on q with r
    route_refuel_comb = {}

    # create list with all unique harbours that can supply any route

    all_harbours=[]
    for harbour in harbours.values():
        all_harbours.append(harbour)

    all_harbours = [x for xs in all_harbours for x in xs]

    all_harbours = list(set(all_harbours))

    for route_key, potential_locations in harbours.items():
        h = []
        # #functioning shortcut, check if any single station is enough (e.g. dist < 2 x r)
        # if r > (path_lengths[route_key]*2):
        #     for facility in harbours[route_key]:
        #         h.append(tuple(facility))
        # else:
        # create all possible station combinations on this path
        for L in range(0, len(potential_locations) + 1):
            for k in itertools.combinations(potential_locations, (L + 1)):
                h.append(k)
        # now add to dict:
        route_refuel_comb[route_key] = h

    # print('route refuel combinations to eval', route_refuel_comb)
    # now check feasibility
    # new master dict to store feasible combinations
    feasible_combinations = {}

    for route_key, route in route_refuel_comb.items():
        feasible_combinations[route_key] = []
        # store path for this route (on which round trip should be feasible)
        harbours_on_route = harbours[route_key]
        # this creates a list with (a, b, c, b, a) if route from a to c via b.
        round_trip = harbours_on_route[:-1] + harbours_on_route[::-1]
        # print(round_trip)

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
                        # print('final dest reached!', current_pos)
                        break
                    # else: maybe feasible, double back route to check!
                    elif current_pos == harbours_on_route[0]:
                        feasible_combinations[route_key].append(combi)
                        # print('final dest reached!', current_pos)
                        break
                else:
                    # print('route unfeasible', current_range-dist)
                    break

    # next: find and remove supersets
    for route_key, combinations in feasible_combinations.items():
        if len(combinations) > 1:
            feasible_combinations[route_key] = get_minimal_subsets(feasible_combinations[route_key])

    # print('feasible combinations', feasible_combinations)
    # Reformat data: create two dicts one with b_qh values and one with g_qhk values
    # first create list of all possible combinations
    unique_combinations = []
    for i in feasible_combinations.values():
        unique_combinations += i

    # print(feasible_combinations)
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
    for node in all_harbours:
        dict_g[node] = []
    # fill second dict for g_qhk
    # print('Combinations:', combinations)
    for route_key, combinations in feasible_combinations.items():
        for combination in combinations:
            dict_g['q'].append(route_key)
            dict_g['h'].append(combination)
            for node in all_harbours:
                # create keys on run for now
                # if not node in dict_g.keys():
                #     dict_g[node] = []
                # print('Node:', node, 'combination:', combination)
                if node in combination:
                    if (node == harbours[route_key][0]) or (node == harbours[route_key][-1]):
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
