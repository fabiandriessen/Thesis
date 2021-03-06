import pandas as pd
import pickle


def flow_computation(df):
    """
    Parameters
    ----------
    df: pd.Dataframe
    This dataframe is compiled using the random_vessel_generator."""
    # ship_data = pickle.load(open("data/flow_comp_factors.p", "rb"))
    ship_data = pd.read_excel('data/ship_types.xlsx')
    ship_data.fillna(0, inplace=True)
    ship_data = dict(zip(ship_data['RWS-class'], ship_data['Factor']))
    pickle.dump(ship_data, open("data/flow_comp_factors.p", "wb"))

    # create dict to store path based values
    flows = {}
    # loop over data frame
    for i in range(len(df)):
        # subset all data ship type data
        a = df.iloc[:, 3:-1]
        # flow is initially 0
        flow = 0
        # add number of ships times specific ship type weighing factor
        for row in a.columns:
            flow += ship_data[row] * a[row][i]
        # store flow, divide by 365 to get daily flow
        flows[(df.origin[i], df.destination[i], df.route_v[i])] = (flow/365)
    return flows

