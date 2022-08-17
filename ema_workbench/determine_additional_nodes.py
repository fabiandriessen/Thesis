import networkx as nx
import pandas as pd
from generate_network_nodes import generate_network


def calc_max_l_adjacent(n, G):
    df_l = nx.to_pandas_edgelist(G)
    df_adj = df_l.loc[(((df_l.source != '8860852') & (df_l.target != '8862614')) |
                       ((df_l.source != '8860852') & (df_l.target != '8861716'))) &
                      ((df_l.target == n) | (df_l.source == n))]
    length_adjacent = 0
    if len(df_adj) > 2:
        df_adj = df_adj.sort_values('length_m').head(2)
        length_adjacent = sum(df_adj.length_m)

    return length_adjacent


def determine_additional_nodes(G, df_h, r):
    df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

    pos_dict = {}
    for node in G.nodes:
        pos_dict[node] = (G.nodes[node]['X'], G.nodes[node]['Y'])

    df_nodes['degree'] = G.degree
    df_nodes['degree'] = df_nodes.degree.apply(lambda x: x[1])

    df_intersections = df_nodes.loc[(df_nodes.degree > 1) & (~df_nodes.n.isin(df_h.harbour_node))]

    df_intersections.n = df_intersections.index

    df_intersections['sum_adjacent'] = df_intersections.apply(lambda x: calc_max_l_adjacent(x.n, G), axis=1)
    df_intersections.sort_values('sum_adjacent', ascending=False, inplace=True)

    additional_intersections = list(df_intersections.loc[df_intersections.sum_adjacent > (0.5 * r)].n)
    print(len(additional_intersections), 'intersections were added')

    return additional_intersections
