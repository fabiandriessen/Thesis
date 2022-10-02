import networkx as nx
import pandas as pd
import pickle
import geopy.distance
pd.options.mode.chained_assignment = None  # default='warn'

def find_nearest_harbour(G, lon, lat, selected):

    df_h = pickle.load(open("data/revised_cleaning_results/harbour_data_100.p", "rb"))
    harbour_nodes = df_h.harbour_node.to_list()

    # extract data
    df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

    # add degree to dataframe and fix n
    df_nodes['degree'] = G.degree
    df_nodes['degree'] = df_nodes.degree.apply(lambda x: x[1])
    df_nodes['n'] = df_nodes.index

    harbours = df_nodes.loc[
        ((df_nodes.n.isin(harbour_nodes)) & (df_nodes.degree > 1)) | (df_nodes.n.str.len() < 4) | (
            df_nodes.n.isin(['8862614', '8860852', '8861819', '8867031', '8867600', '8860933', '8863288'])) | (
            df_nodes.n.isin(selected))]
    x = lon
    y = lat
    dev = 0.5

    # find nodes within deviation
    # select nodes near
    selection = harbours.loc[(harbours.X.between(x - dev, x + dev)) & (harbours.Y.between(y - dev, y + dev))]

    # in some areas there are very few nodes,
    # therefore iteratively increase range to look for nodes until at least one is found

    while len(selection) == 0:
        dev += 0.5
        selection = harbours.loc[
            (harbours.X.between(x - dev, x + dev)) & (harbours.Y.between(y - dev, y + dev))].index

    selection['dist'] = selection.apply(lambda m: (geopy.distance.geodesic((m.X, m.Y), (lon, lat))), axis=1)

    return selection.dist.min()


def additional_intersections(G, no_selected_intersections):

    df_h = pickle.load(open("data/revised_cleaning_results/harbour_data_100.p", "rb"))
    harbour_nodes = df_h.harbour_node.to_list()

    # extract data
    df_nodes = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')

    # add degree to dataframe and fix n
    df_nodes['degree'] = G.degree
    df_nodes['degree'] = df_nodes.degree.apply(lambda x: x[1])
    df_nodes['n'] = df_nodes.index

    # potential included intersections are nodes with degree higher than 2 and not already included nodes
    intersection_df = df_nodes.loc[(df_nodes.degree > 2) & (~df_nodes.n.isin(harbour_nodes))]

    selected = []
    for i in range(no_selected_intersections):
        intersection_df['dist_nearest_harbour'] = intersection_df.apply(
            lambda m: (find_nearest_harbour(G, m.X, m.Y, selected)), axis=1)
        faraway = intersection_df.sort_values('dist_nearest_harbour', ascending=False).head(1).n.to_list()
        selected += faraway
    print(len(selected), 'intersections were added')

    return selected


