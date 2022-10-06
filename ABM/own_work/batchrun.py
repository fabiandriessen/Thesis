from multiprocessing import freeze_support
from mesa.batchrunner import batch_run
from model import VesselElectrification
import pickle
import numpy as np

pick_from = np.linspace(0, 1000000, 1000001)
seeds = np.random.choice(a=pick_from, size=100, replace=False)
seeds = list(seeds)
seeds = [round(i) for i in seeds]
# print(seeds)
parameters = {}
if __name__ == '__main__':
    freeze_support()
    result = batch_run(VesselElectrification, iterations=1, parameters={'seed': seeds},
                       data_collection_period=(60 * 24 * 8), max_steps=(60 * 24 * 8), number_processes=17)
    pickle.dump(result, open('data/batch_run_result_2_5000.p', 'wb'))
