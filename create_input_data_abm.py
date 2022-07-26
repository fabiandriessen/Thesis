import pandas as pd
import pickle
import networkx as nx
import numpy as np


def cs_harbour_or_both(col1, col2):
    to_r = col2
    if col1 == 0:
        if col2 == 'inserted_node':
            to_r = 'charging_station'
        else:
            to_r = 'harbour_with_charging'
    return to_r


def create_input_data_abm(G, paths, non_zero_flows, optimal_facilities):
    df_h = pickle.load(open("data/revised_cleaning_results/harbour_data_100.p", "rb"))
    nodes = []
    for route in non_zero_flows.keys():
        nodes += paths[route]

    nodes = list(set(nodes))
    # %%
    df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    # %%
    df_nodes = df_nodes.loc[df_nodes.index.isin(nodes)]
    # %%
    df_nodes.fillna(0, inplace=True)
    df_nodes['model_type'] = df_nodes.n.apply(
        lambda x: "harbour" if ((len(str(x)) > 4) and (x in df_h.harbour_node.unique())) else x)
    df_nodes.model_type = df_nodes.model_type.apply(lambda x: "inserted_node" if x == 0 else x)
    df_nodes.model_type = df_nodes.model_type.apply(lambda x: "intermediate_node" if str(x).isdigit() else x)
    df_nodes['index1'] = df_nodes.index
    df_nodes['charging_stations'] = df_nodes.index1.apply(
        lambda x: optimal_facilities[x] if (x in df_h.harbour_node.unique()) or (len(str(x)) == 3) else 0)
    df_nodes.model_type = df_nodes.apply(lambda x: cs_harbour_or_both(x.charging_stations, x.model_type), axis=1)
    
    df_nodes.drop(columns=['index1', 'n', 'geometry', 'Wkt'], inplace=True)
    df_nodes.reset_index(inplace=True)
    df_nodes.rename(columns={'index': 'node_id'}, inplace=True)

    df_nodes['source'] = np.nan
    df_nodes['target'] = np.nan
    df_nodes['length_m'] = np.nan

    df_links = nx.to_pandas_edgelist(G)
    df_links = df_links.loc[:, ['source', 'target', 'length_m']]

    df_links['X'] = np.nan
    df_links['Y'] = np.nan
    df_links['node_id'] = np.nan
    df_links['charging_stations'] = np.nan
    df_links['model_type'] = 'link'
    df_abm = pd.concat([df_links, df_nodes])

    return df_abm
