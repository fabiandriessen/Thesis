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

    # prepare dataframe
    df = df.groupby(by=['origin', 'destination', 'route_v']).sum().reset_index().drop(columns=['hour'])

    df = df[['origin', 'destination', 'trip_count', 'M12', 'M8', 'BII-6b', 'M10', 'BIIa-1', 'M9', 'BII-6l',
                       'C3b', 'BII-4', 'M7', 'M6', 'BIIL-1', 'C3l', 'M5', 'M11', 'BI', 'M3', 'M2', 'M1', 'BII-1',
                       'BII-2b', 'M4', 'B03', 'C4', 'B04', 'M0', 'C2l', 'BII-2L', 'B02', 'C1b', 'C2b', 'B01', 'C1l',
                       'route_v']]

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

