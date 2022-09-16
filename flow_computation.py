import pandas as pd
import pickle


def flow_computation(df, r, path_lengths, individual_speed=True):
    """
    Parameters
    ----------
    df: pd.Dataframe
    This dataframe is compiled using the random_vessel_generator.

    r: int
    The range of a ship that should be considered, this is used calculate the battery sizes of the various ships.

    individual_speed: Boolean
    If true, use individual speeds to calculate battery sizes, else use one value for all the ships.
    """
    # read in ship data
    ship_data = pd.read_excel('data/ship_types.xlsx')

    # determine battery size using either individual or mean speed, speed must be converted to m/h from m/s.
    if individual_speed:
        ship_data['battery_size'] = ship_data.apply(lambda x: (60000 / (x.speed_loaded * 3.6 * 1000)) * x.P_average,
                                                    axis=1)
    else:
        ship_data['battery_size'] = ship_data.apply(lambda x: (r/(ship_data.speed_loaded.mean() * 3.6 * 1000))
                                                    * x.P_average, axis=1)

    # determine factors based on battery size
    ship_data['Factor'] = ship_data.battery_size.apply(lambda n: (n / ship_data.battery_size[0]))
    ship_data = dict(zip(ship_data['RWS-class'], ship_data['battery_size']))

    # prepare dataframe
    df = df.groupby(by=['origin', 'destination', 'route_v']).sum().reset_index().drop(columns=['hour'])

    # create dict to store path based values
    flows = {}

    # loop over all unique routes
    for _, row in df.iterrows():
        q = (row['origin'], row['destination'], row['route_v'])
        flow = 0
        # sum flow factor for each ship type
        for ship_type, battery_size in ship_data.items():
            flow += row[ship_type] * battery_size
        flows[q] = (flow / 365) * (path_lengths[q]/r)

    return flows  # flows are in kWh now!
