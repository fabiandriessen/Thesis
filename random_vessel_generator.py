import pandas as pd
import numpy as np


def random_vessel_generator(df_prob, load=1):
    """ Function to generate random vessel data.
    Parameters
    ----------
    df_prob: pd.DataFrame
    This dataframe must contain historical travel data
    load: float
    This float is equal to the simulated occupation of the network and may be read as the part of the traffic that is
    electric.
    """
    df_prob = df_prob.loc[df_prob.trip_count != 0]
    df_prob.reset_index(inplace=True, drop=True)
    df_prob = df_prob.fillna(0)

    # create dict to store random prob based values later on
    main_dict = {i: [] for i in df_prob.columns}

    # copy origin and destination from original df
    main_dict['origin'] = []
    main_dict['destination'] = []
    main_dict['route_v'] = []
    main_dict['hour'] = []
    main_dict['trip_count'] = []

    # fill nan values and create type list
    type_list = list(df_prob.iloc[:, 4:-1])
    for i in range(24):
        df_temp = df_prob.loc[df_prob.hour == i]
        df_temp = df_temp.reset_index(drop=True)
        total = df_temp.trip_count.sum()
        prob_r = [i / total for i in df_temp.trip_count]
        to_pick_r = list(np.arange(0, len(df_temp)))
        count_r = total * load
        # randomly draw ODs
        rand_routes = np.random.choice(a=to_pick_r, size=round(count_r), replace=True, p=prob_r)
        unique_r, counts_r = np.unique(rand_routes, return_counts=True)
        temp_type_dict_r = dict(zip(unique_r, counts_r))

        for pair in df_temp.index:
            # store trip count data previous draw
            if pair in temp_type_dict_r.keys():
                main_dict['trip_count'].append(temp_type_dict_r[pair])
                main_dict['origin'].append(df_temp.origin[pair])
                main_dict['destination'].append(df_temp.destination[pair])
                main_dict['route_v'].append(df_temp.route_v[pair])
                main_dict['hour'].append(i)
                # print(main_dict['trip_count'])
                # determine parameters to randomly draw ship types for each OD pair
                prob = list(df_temp.iloc[pair, 4:-1].values / df_temp.iloc[pair, 4:-1].sum())
                to_pick = type_list
                # goes wrong now because may be zero ...
                count = temp_type_dict_r[pair]

                # randomly draw ship types for each OD pair
                rand_vessels = np.random.choice(a=to_pick, size=round(count), replace=True, p=prob)
                unique, counts = np.unique(rand_vessels, return_counts=True)
                temp_type_dict = dict(zip(unique, counts))
                for key in type_list:
                    if key in temp_type_dict.keys():
                        main_dict[key].append(temp_type_dict[key])
                    else:
                        main_dict[key].append(0)

    df_return = pd.DataFrame.from_dict(main_dict)

    return df_return
