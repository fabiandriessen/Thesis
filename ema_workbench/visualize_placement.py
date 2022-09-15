import matplotlib.pyplot as plt
import networkx as nx
# import osmnx as ox
import geopandas as gpd
# ox.config(log_console=True, use_cache=True)


def visualize_placement(G, OD, optimal_facilities, non_zero_flows, df_h, paths, unused=True):
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
    # create pos dict
    pos_dict = {}
    for node in G.nodes:
        pos_dict[node] = (G.nodes[node]['X'], G.nodes[node]['Y'])

    # Define new graph H with only nodes and edges in available routes
    node_list = []

    for key, items in non_zero_flows.items():
        node_list += paths[key]

    img = plt.imread("data/NL_map.png")
    # sub graph with supported nodes, only keep unique nodes
    node_list = list(set(node_list))
    H = G.subgraph(node_list)

    # # sub graph with all nodes in unsupported routes
    # K = G.subgraph(other_nodes)
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

    H_fuel = G.subgraph(fuel_stations)
    # print(H_fuel.nodes)
    # now draw, first setup grid
    # fig, ax = plt.subplots(dpi=200, figsize=(2048/200, 1817/200))
    # fig, ax = plt.subplots(dpi=200, figsize=(3.778074*2*0.8, 3.9834440000000058*2))
    fig, ax = plt.subplots(dpi=200)

    # all edges related to route
    nx.draw_networkx_edges(G, pos_dict, width=2, ax=ax, label='Unsupported paths')
    nx.draw_networkx_edges(H, pos=pos_dict, width=2, edge_color='red', ax=ax, label='Supported paths')

    # draw origin and destination nodes
    # nx.draw_networkx_nodes(G, pos_dict, origins, node_color='green', node_size=100, alpha=0.5)
    # nx.draw_networkx_nodes(G, pos_dict, destinations, node_color='yellow', node_size=100, alpha=0.5)

    # fuel station nodes in red with label = number of fuel stations placed
    nx.draw_networkx_nodes(H_fuel, pos_dict, node_color='red', node_size=30, alpha=1, ax=ax, label='Charging stations')
    nx.draw_networkx_labels(H_fuel, pos_dict, labels=nx.get_node_attributes(H_fuel, 'number_CS'), ax=ax, font_size=6)

    # unused potential fuel station locations in blue (if argument = True)
    if unused:
        nx.draw_networkx_nodes(G, pos_dict, other_ks, node_color='blue', alpha=1, node_size=30, ax=ax,
                               label='Unused locations')
    plt.legend(fontsize=6)

    x1, x2, y1, y2 = 3.566619, 6.944693, 50.860830, 53.444274
    z = 0.53
    ax.imshow(img, extent=[x1-z, x2+z, y1-z+0.3, y2+z-0.3])
    return H_fuel

