from multiprocessing import freeze_support
from mesa.batchrunner import batch_run
from model import VesselElectrification
import pickle


def run_abm(paths, G, df_ivs, df_abm, optimal_flows):
    if __name__ == '__main__':
        freeze_support()
        result = batch_run(VesselElectrification, iterations=1, parameters={'seed': [78393, 206055, 59784, 306260,
                                                                                     972222, 36622, 282095, 507368],
                                                                            "paths": paths,
                                                                            "G": G,
                                                                            "df_ivs": df_ivs,
                                                                            "df_abm": df_abm,
                                                                            "optimal_flows": optimal_flows},
                           data_collection_period=(60 * 24 * 8), max_steps=(60 * 24 * 8), number_processes=None)

        return result
