import pandas as pd
import numpy as np


def random_vessel_generator(df_prob, load):

    # create dict to store random prob based values later on
    main_dict = {i: [] for i in df_prob.columns}

    # copy origin and destination from original df
    main_dict['origin'] = list(df_prob.origin)
    main_dict['destination'] = list(df_prob.destination)
    main_dict['route_v'] = list(df_prob.route_v)
    main_dict['trip_count'] = []

    # fill nan values and create type list
    df_prob.fillna(0, inplace=True)
    type_list = list(df_prob.iloc[:, 3:-1])
    total = df_prob.trip_count.sum()

    # determine parameters randomly drawing ODs
    prob_r = [i / total for i in df_prob.trip_count]

    to_pick_r = list(np.arange(0, len(df_prob)))
    count_r = total * load

    # randomly draw ODs
    rand_routes = np.random.choice(a=to_pick_r, size=round(count_r), replace=True, p=prob_r)
    unique_r, counts_r = np.unique(rand_routes, return_counts=True)
    temp_type_dict_r = dict(zip(unique_r, counts_r))

    # for each OD pair
    for pair in range(len(df_prob)):

        # store trip count data previous draw
        if pair in temp_type_dict_r.keys():
            main_dict['trip_count'].append(temp_type_dict_r[pair])
        # chance that nothing is drawn, then append 0
        else:
            main_dict['trip_count'].append(0)

        # determine parameters to randomly draw ship types for each OD pair
        prob = list(df_prob.iloc[pair, 3:-1].values/df_prob.iloc[pair, 3:-1].sum())
        to_pick = type_list
        count = main_dict['trip_count'][pair]

        # randomly draw ship types for each OD pair
        rand_vessels = np.random.choice(a=to_pick, size=round(count), replace=True, p=prob)
        unique, counts = np.unique(rand_vessels, return_counts=True)
        temp_type_dict = dict(zip(unique, counts))

        # append amount of random generated vessels right dict list
        for key in type_list:
            if key in temp_type_dict.keys():
                main_dict[key].append(temp_type_dict[key])
            else:
                main_dict[key].append(0)

    # now make dict
    df_return = pd.DataFrame.from_dict(main_dict)

    return df_return
