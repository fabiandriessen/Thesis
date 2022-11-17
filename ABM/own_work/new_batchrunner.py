from multiprocessing import freeze_support
from mesa.batchrunner import batch_run
from model_new_v import VesselElectrification
import pickle
import numpy as np
from create_input_data_abm import create_input_data_abm
import pandas as pd


def create_key(o, d, r_v):
    key1 = (o, d, r_v)
    return key1


# pick_from = np.linspace(0, 1000000, 1000001)
# seeds = np.random.choice(a=pick_from, size=100, replace=False)
# seeds = list(seeds)
# seeds = [round(i) for i in seeds]
# seeds = [801113, 381219, 170962, 548806, 205426, 136991, 603019, 128545, 541820, 139481, 87277, 579298, 203921,
#          918913, 874827, 915849, 354585, 964632, 61386, 829384]
# seeds = print(seeds)
# #%%
seeds = [943174, 59109, 940447, 894642, 58457, 287378, 904056, 148406, 251946, 606491, 55482, 111463, 324879, 174330,
         211568, 658685, 578467, 767133, 4652, 577384, 424649, 353559, 396676, 87653, 638938, 49198, 725790, 226624,
         759283, 512562, 306832, 673778, 277157, 814521, 690540, 809293, 498009, 170028, 345184, 101340, 518082, 441369,
         780080, 153571, 529393, 778076, 716417, 344652, 256323, 629099, 748682, 675818, 154773, 273247, 649182, 335143,
         343568, 201145, 380805, 609979, 141007, 695335, 303535, 23190, 876857, 453751, 102408, 746183, 497756, 89929,
         955043, 924879, 752243, 359665, 644097, 391549, 912429, 908238, 557321, 193006, 522392, 852871, 29309, 878055,
         673785, 600049, 699576, 245503, 674099, 902826, 514899, 358866, 146967, 785935, 319769, 895062, 927804, 613149,
         490007, 247556]

# import dataframe with input parameters for each run
df_9scenarios = pd.DataFrame(pickle.load(open("data/df_all_without_clean.p", "rb")))
G = pickle.load(open('data/network.p', 'rb'))
paths = pickle.load(open('data/paths.p', 'rb'))

for i, row in df_9scenarios.iterrows():
    print(row['c'], row['r'], i)
    df_abm = create_input_data_abm(G, paths, row['non_zero_flows'], row['optimal_facilities'])
    # configure df random for abm
    df_random = pickle.load(open('data/df_random.p', 'rb'))
    df_random['key'] = df_random.apply(lambda x: create_key(x.origin, x.destination, x.route_v), axis=1)
    df_random = df_random.loc[df_random.key.isin(row['non_zero_flows'].keys())]
    df_random = df_random.loc[df_random.trip_count != 0]
    pickle.dump(df_random, open('data/inputs/df_random_batch' + str(i) + '.p', 'wb'))
    df_abm.to_csv('data/inputs/df_abm_batch' + str(i) + '.csv')
    pickle.dump(row['non_zero_flows'], open('data/inputs/non_zero_flows_batch' + str(i) + '.p', 'wb'))
    # params = {'c': row['c'], 'r': row['r'], 'run': i, 'seed': seeds}
    #
    # if __name__ == '__main__':
    #     freeze_support()
    #     result = batch_run(VesselElectrification,
    #                        iterations=1,
    #                        parameters=params,
    #                        data_collection_period=(60 * 24 * 8),
    #                        max_steps=(60 * 24 * 8),
    #                        number_processes=17,
    #                        display_progress=True)
    #
    #     pickle.dump(result,
    #                 open('results_without/batch_9scenarios' + str(row['r']) + str(row['c']) + str(row['m']) + '.p', 'wb'))
