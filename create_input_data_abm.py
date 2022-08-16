import pandas as pd
import pickle
import networkx as nx
import numpy as np

def cs_harbour_or_both(col1, col2):
    to_r = col2
    if col1 != 0:
        if col2 == 'inserted_node':
            to_r = 'charging_station'
        else:
            to_r = 'harbour_with_charging'
    return to_r


def gen_link_name(col1, col2):
    return tuple([col1, col2])


def find_middle_x(val1, val2, df_nodes):
    x1 = df_nodes.loc[df_nodes.name == val1].X.item()
    x2 = df_nodes.loc[df_nodes.name == val2].X.item()
    return (x1+x2)/2


def find_middle_y(val1, val2, df_nodes):
    y1 = df_nodes.loc[df_nodes.name == val1].Y.item()
    y2 = df_nodes.loc[df_nodes.name == val2].Y.item()
    return (y1+y2)/2


def create_input_data_abm(G, paths, non_zero_flows, optimal_facilities):
    df_h = pickle.load(open("data/revised_cleaning_results/harbour_data_100.p", "rb"))
    nodes = []
    for route in non_zero_flows.keys():
        nodes += paths[route]

    G = G.subgraph(nodes)

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
        lambda x: optimal_facilities[x] if x in optimal_facilities.keys() else 0)

    df_nodes.model_type = df_nodes.apply(lambda x: cs_harbour_or_both(x.charging_stations, x.model_type), axis=1)

    df_nodes.drop(columns=['index1', 'n', 'geometry', 'Wkt'], inplace=True)
    df_nodes.reset_index(inplace=True)
    df_nodes.rename(columns={'index': 'name'}, inplace=True)

    df_nodes['source'] = np.nan
    df_nodes['target'] = np.nan
    df_nodes['length_m'] = np.nan

    df_links = nx.to_pandas_edgelist(G)
    df_links = df_links.loc[:, ['source', 'target', 'length_m']]
    df_links['X'] = df_links.apply(lambda x: find_middle_x(x.source, x.target, df_nodes), axis=1)
    df_links['Y'] = df_links.apply(lambda x: find_middle_y(x.source, x.target, df_nodes), axis=1)

    df_links['name'] = df_links.apply(lambda x: gen_link_name(x.source, x.target), axis=1)
    df_links['charging_stations'] = np.nan
    df_links['model_type'] = 'link'
    df_abm = pd.concat([df_links, df_nodes])
    df_abm.reset_index(drop=True, inplace=True)
    df_abm.reset_index(inplace=True)
    df_abm.rename(columns={'index': 'id'}, inplace=True)
    df_abm.id = df_abm.id.apply(lambda x: x + 10000)

    return df_abm
