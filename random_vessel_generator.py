import pandas as pd
import numpy as np


def random_vessel_generator(df_prob):
    ship_data = pd.read_excel('data/ship_types.xlsx')
    ship_data.fillna(0, inplace=True)
    ship_data = dict(zip(ship_data['RWS-class'], ship_data['Factor']))

    df_prob.fillna(0, inplace=True)
    # create dict to store random prob based values later on
    main_dict = {i: [] for i in df_prob.columns}

    # loop over all rows of the probability dataframe
    for i in range(len(df_prob)):
        # copy origin, destination and count from original df
        for x in df_prob.columns[:3]:
            main_dict[x].append(df_prob.iloc[i, :][x])

        # find probability, items to pick, and the number of vessels to generate in total
        prob = list(df_prob.iloc[i, 3:-1].values)
        to_pick = list(df_prob.columns)
        to_pick = to_pick[3:-1]
        count = df_prob['trip_count'][i]
        # print(to_pick, count, prob)
        # generate random vessels
        rand_vessels = np.random.choice(a=to_pick, size=round(count), replace=True, p=prob)
        unique, counts = np.unique(rand_vessels, return_counts=True)
        temp_dict = dict(zip(unique, counts))

        # append amount of random generated vessels right dict list
        for key in list(main_dict.keys())[3:]:
            if key in temp_dict.keys():
                main_dict[key].append(temp_dict[key])
            else:
                main_dict[key].append(0)
        # now make dict
        df_return = pd.DataFrame.from_dict(main_dict)
        df_return['route_v'] = df_prob['route_v']
    return df_return
