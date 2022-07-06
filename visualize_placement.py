import matplotlib.pyplot as plt
import networkx as nx


def visualize_placement(G, OD, optimal_facilities, df_h, paths, unused=True):
    """Function to draw network, routes and charging stations, based on a graph G, an origin destination list,
    and a dictionary with the facilities to visualize.

        Parameters
    ----------
    G : NetworkX graph
        must include all origins, destinations and any nodes where a refueling station may be placed.
    OD: dict
        A list of travel data within network G, travel data from A-B and from B-A should be summed up and
        entered as either one of them.
        example input:
        [(node_1, node_2, flow_12),(node_1, node_3, flow_13),(node_2, node_3, flow_23)]
    optimal_facilities: dict
        A dictionary that contains the nodes that are potential charging station locations as a key, and the number of
        charging stations that are placed at a node as an entry (N+).
    df_h: pd.DataFrame
        This dataframe as generated in revised_network_cleaning.ipynb
        """

    # Define new graph H with only nodes and edges in routes
    node_list = []
    origins = []
    destinations = []

    for (origin, destination, version), flow in OD.items():
        origins.append(df_h.loc[df_h.harbour_code == origin]['harbour_node'].values[0])
        destinations.append(df_h.loc[df_h.harbour_code == destination]['harbour_node'].values[0])
        node_list.append(paths[(origin, destination, version)])

    expanded_node_list = [x for xs in node_list for x in xs]
    node_list = list(set(expanded_node_list))

    H = G.subgraph(node_list)

    # create pos dict
    pos_dict = {}
    for node in node_list:
        pos_dict[node] = (G.nodes[node]['X'],G.nodes[node]['Y'])

    # make two lists: one with used locations and one with unused locations
    other_ks = []
    fuel_stations = []

    # fill lists
    for key, number_of_stations in optimal_facilities.items():
        if number_of_stations > 0:
            fuel_stations.append(key)
            G.nodes[key]['number_CS'] = number_of_stations
        else:
            other_ks.append(key)

    H_fuel = H.subgraph(fuel_stations)

    # now draw, first setup grid
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)

    # all edges related to route
    nx.draw_networkx_edges(H, pos=pos_dict, width=2)

    # draw origin and destination nodes
    nx.draw_networkx_nodes(G, pos_dict, origins, node_color='green', node_size=100, alpha=0.5)
    nx.draw_networkx_nodes(G, pos_dict, destinations, node_color='yellow', node_size=100, alpha=0.5)

    # fuel station nodes in red with label = number of fuel stations placed
    nx.draw_networkx_nodes(H_fuel, pos_dict, node_color='red', alpha=0.5)
    nx.draw_networkx_labels(H_fuel, pos_dict, labels=nx.get_node_attributes(H_fuel, 'number_CS'))

    # unused potential fuel station locations in blue (if argument = True)
    if unused:
        nx.draw_networkx_nodes(G, pos_dict, other_ks, node_color='blue', alpha=0.5)
    plt.show()
