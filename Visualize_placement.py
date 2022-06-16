import matplotlib.pyplot as plt
import networkx as nx


def visualize_placement(G, OD_list, optimal_facilities, flow_allocation):
    # Draw network, routes and charging stations

    # Define new graph H with only nodes and edges in routes

    node_list =[]
    for origin, destination, flow in OD_list:
        for node in nx.dijkstra_path(G, origin, destination, weight='length_m'):
            if node not in node_list:
                node_list.append(node)

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
        else:
            other_ks.append(key)

    # now draw: all edges related to route, fuel_station nodes, and unused potential locations
    nx.draw_networkx_edges(H, pos = pos_dict)

    nx.draw_networkx_nodes(G,pos_dict, fuel_stations, node_color='red')
    nx.draw_networkx_nodes(G,pos_dict,other_ks, node_color='blue')
    plt.show()