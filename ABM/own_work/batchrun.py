from multiprocessing import freeze_support
from mesa.batchrunner import batch_run
from model import VesselElectrification
import pickle

parameters = {}
if __name__ == '__main__':
    freeze_support()
    result = batch_run(VesselElectrification, iterations=1, parameters={'seed': range(1000, 1010)},
                       data_collection_period=(60 * 24 * 8), max_steps=(60 * 24 * 8), number_processes=None)
    pickle.dump(result, open('data/batch_run_result.p', 'wb'))
