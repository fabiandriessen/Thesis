import pandas as pd
import pickle
import networkx as nx


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
    df_nodes['node_type'] = df_nodes.n.apply(
        lambda x: "harbour" if ((len(str(x)) > 4) and (x in df_h.harbour_node.unique())) else x)
    df_nodes.node_type = df_nodes.node_type.apply(lambda x: "inserted_node" if x == 0 else x)
    df_nodes.node_type = df_nodes.node_type.apply(lambda x: "intermediate_node" if str(x).isdigit() else x)
    df_nodes['index1'] = df_nodes.index
    df_nodes['charging_stations'] = df_nodes.index1.apply(lambda x: optimal_facilities[x] if
    (x in df_h.harbour_node.unique()) or (len(str(x)) == 3) else 0)
    df_nodes.drop(columns=['index1', 'n', 'geometry', 'Wkt'], inplace=True)

    df_links = nx.to_pandas_edgelist(G)
    df_links = df_links.loc[:, ['source', 'target', 'length_m']]

    return df_nodes, df_links
