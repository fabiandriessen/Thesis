import networkx as nx
import pandas as pd
import numpy as np
import math


def generate_network(G, paths, r):

    """This is a function to generate a network with n additional nodes to minimize the maximum link length. A node is
    placed in the middle of the longest node, if a node already has been split, the original node will be split in
    three equally long edges.
    Parameters
    ----------
    G: nx.Graph
        Networkx graph that should contain only direct connections between the essential nodes, may be generated using
        jupyter notebook 4.
    paths: dict
        This dictionary should contain all the paths between the various origins and destinations as generated in
        notebook 3.
    r: int
        Range of a vessel.
     """

    # retrieve data from G
    pos_dict = {}
    for node in G.nodes:
        pos_dict[node] = (G.nodes[node]['X'], G.nodes[node]['Y'])

    # set new edge attribute split initially to 0 for all edges
    nx.set_edge_attributes(G, 0, 'split')

    # first inserted node gets ID 100 and from there upwards
    id_count = 100
    inserted = []

    df_links = nx.to_pandas_edgelist(G)
    df_links = df_links.loc[((df_links.source != '8860852') & (df_links.target != '8862614')) | (
            (df_links.source != '8860852') & (df_links.target != '8861716'))]

    for i in range(1000):
        if math.ceil(max(df_links.length_m)) <= (r * 0.5):
            print("There were", len(inserted), "nodes added, the longest remaining link is now:",
                  df_links.length_m.max())
            break

        # update dataframes
        df_links = nx.to_pandas_edgelist(G)
        df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

        # find the longest link source and origin
        # find the longest link source and origin, except for the two links that cross the sea
        df_links = df_links.loc[((df_links.source != '8860852') & (df_links.target != '8862614')) | (
                    (df_links.source != '8860852') & (df_links.target != '8861716'))]
        df_links.reset_index(inplace=True, drop=True)
        to_split = df_links.loc[df_links.length_m == max(df_links.length_m)]
        to_split.reset_index(inplace=True, drop=True)
        # identify source/targets points
        points = list(to_split.loc[(to_split.source.str.len() > 5)].source.values) + list(
            to_split.loc[(to_split.target.str.len() > 5)].target.values)

        org_source = points[0]
        org_target = points[1]

        # determine original length, also works if link has been split earlier
        original_length = 0
        for j in to_split.length_m:
            original_length += j

        org_source_data = df_nodes.loc[df_nodes.n == org_source]
        org_target_data = df_nodes.loc[df_nodes.n == org_target]

        # if never split: split in two, if split once: split in three, etc.
        split_in = round(to_split.split[0] + 2)

        # find new X and Y positions
        x_set = np.linspace(org_source_data.X.values[0], org_target_data.X.values[0], (split_in + 1))
        y_set = np.linspace(org_source_data.Y.values[0], org_target_data.Y.values[0], (split_in + 1))

        # remove old link(s)
        for j in to_split.index:
            G.remove_edge(to_split.source[j], to_split.target[j])
        # if split before, remove earlier inserted points
        points_to_remove = list(to_split.loc[(to_split.source.str.len() < 5)].source.values) + list(
            to_split.loc[(to_split.target.str.len() < 5)].target.values)
        if points_to_remove:
            for j in set(points_to_remove):
                G.remove_node(j)
                inserted.remove(j)

        # add nodes, except for outsides that already exist
        added_ids = []
        for index_pos, j in enumerate(x_set):
            if (index_pos != 0) and (index_pos != (len(x_set) - 1)):
                G.add_node(str(id_count), X=x_set[index_pos], Y=y_set[index_pos])
                added_ids.append(str(id_count))
                inserted.append(str(id_count))
                id_count += 1

        # finally, add edges
        nodes_sequence = [org_source] + added_ids + [org_target]
        for j in range(len(nodes_sequence) - 1):
            G.add_edge(nodes_sequence[j], nodes_sequence[j + 1], length_m=(original_length / split_in),
                       split=int(to_split.split[0] + 1))

        # redetermine df
        df_links = nx.to_pandas_edgelist(G)
        df_links = df_links.loc[((df_links.source != '8860852') & (df_links.target != '8862614')) | (
                (df_links.source != '8860852') & (df_links.target != '8861716'))]

        # break out of loop if longest link is small enough
        if math.ceil(max(df_links.length_m)) <= (r * 0.5):
            print("There were", len(inserted), "nodes added, the longest remaining link is now:",
                  df_links.length_m.max())
            break

    # fix insertion of additional nodes in route!
    for route, path in paths.items():
        new_route = [path[0]]
        for node_index in range(len(path) - 1):
            p = nx.dijkstra_path(G, path[node_index], path[node_index + 1])
            new_route += p[1:]
        paths[route] = new_route

    return G, paths, inserted
