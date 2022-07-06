import pandas as pd


def flow_computation(df):

    ship_data = pd.read_excel('data/ship_types.xlsx')
    ship_data.fillna(0, inplace=True)
    ship_data = dict(zip(ship_data['RWS-class'], ship_data['Factor']))

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

