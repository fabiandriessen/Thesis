from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Harbour, HarbourChargingStation, ChargingStation, Link, Intersection, Vessel
import pandas as pd
from mesa.datacollection import DataCollector
from collections import defaultdict
import networkx as nx
import pickle
import numpy as np
import random
import math


def get_waited_at(agent):
    return agent.waited_at if isinstance(agent, Vessel) else np.nan


def get_key(agent):
    return agent.route_key if isinstance(agent, Vessel) else np.nan


def get_departed_at(agent):
    return agent.generated_at_step if isinstance(agent, Vessel) else np.nan


def get_battery_size(agent):
    return agent.battery_size if isinstance(agent, Vessel) else np.nan


def get_battery_level(agent):
    return agent.charge if isinstance(agent, Vessel) else np.nan


def get_generation_time(agent):
    return agent.generated_at_step if isinstance(agent, Vessel) else np.nan


def get_removal_time(agent):
    return agent.removed_at_step if isinstance(agent, Vessel) else np.nan


def get_combi(agent):
    return agent.combi if isinstance(agent, Vessel) else np.nan


def get_departed_from(agent):
    return agent.generated_by.unique_id if isinstance(agent, Vessel) else np.nan


def get_vessel_status(agent):
    if isinstance(agent, Vessel):
        if agent.state == Vessel.State.DRIVE:
            return 'driving'
        else:
            if agent.inline:
                return ['inline', agent.location.unique_id]
            else:
                return ['charging', agent.location.unique_id]
    else:
        return np.nan


def get_station_status(agent):
    if isinstance(agent, ChargingStation) or isinstance(agent, HarbourChargingStation):
        return len(agent.currently_charging) / agent.modules
    else:
        return np.nan


def get_cs(agent):
    if isinstance(agent, ChargingStation) or isinstance(agent, HarbourChargingStation):
        return agent.modules
    else:
        return np.nan


def get_cs_occupation(agent):
    if isinstance(agent, ChargingStation) or isinstance(agent, HarbourChargingStation):
        if (agent.users > 0) and (agent.steps_measuring > 0):
            avg_occupation = agent.users / agent.steps_measuring
            return avg_occupation
        else:
            return 0
    else:
        return np.nan


def get_max_occupation(agent):
    if isinstance(agent, ChargingStation) or isinstance(agent, HarbourChargingStation):
        return agent.max_occupation


def get_cs_waiting_line(agent):
    if isinstance(agent, ChargingStation) or isinstance(agent, HarbourChargingStation):
        if (agent.waiters > 0) and (agent.steps_measuring > 0):
            avg_line_length = agent.waiters / agent.steps_measuring
            return avg_line_length
        else:
            return 0
    else:
        return np.nan


def get_max_line_length(agent):
    if isinstance(agent, ChargingStation) or isinstance(agent, HarbourChargingStation):
        return agent.max_line_l


# ---------------------------------------------------------------
def set_lat_lon_bound(lat_min, lat_max, lon_min, lon_max, edge_ratio=0.02):
    """
    Set the HTML continuous space canvas bounding box (for visualization)
    give the min and max latitudes and Longitudes in Decimal Degrees (DD)

    Add white borders at edges (default 2%) of the bounding box
    """

    lat_edge = (lat_max - lat_min) * edge_ratio
    lon_edge = (lon_max - lon_min) * edge_ratio

    x_max = lon_max + lon_edge
    y_max = lat_min - lat_edge
    x_min = lon_min - lon_edge
    y_min = lat_max + lat_edge
    return y_min, y_max, x_min, x_max


# ---------------------------------------------------------------
class VesselElectrification(Model):
    """
    The main (top-level) simulation model

    One tick represents one minute; this can be changed
    but the distance calculation need to be adapted accordingly

    Class Attributes:
    -----------------
    step_time: int
        step_time = 1 # 1 step is 1 min

    path_ids_dict: default-dict
        Key: (origin, destination)
        Value: the shortest path (Infra component IDs) from an origin to a destination

        Only straight paths in the Demo are added into the dict;
        when there is a more complex network layout, the paths need to be managed differently

    sources: list
        all sources in the network

    sinks: list
        all sinks in the network

    """

    step_time = 1

    def __init__(self, c, r, run, seed, x_max=500, y_max=500, x_min=0, y_min=0):
        self.seed = seed
        np.random.seed = self.seed
        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = pickle.load(open('data/final_paths.p', 'rb'))  # paths for the network inc. any added nodes
        self.space = None
        self.G = pickle.load(open('data/network_cleaned_final.p', 'rb'))  # the actual network as generated inc. any added nodes
        # self.ivs_data = input_df
        self.ivs_data = pickle.load(
            open('data/inputs/df_random_batch' + str(run) + '.p', 'rb'))  # random dataset to base generation on
        # self.data = df_abm
        self.data = pd.read_csv(
            'data/inputs/df_abm_batch' + str(run) + '.csv')  # all information regarding different links, harbours, etc.
        self.intermediate_nodes = []
        self.harbours = []
        self.harbours_with_charging = []
        self.links = []
        self.charging_stations = []
        self.inserted_nodes = []
        self.hour = 0
        self.charging_station_capacity = c  # max charging power in KWh
        self.range = r * 1000  # range in meters

        # self.path_lengths = pickle.load(open('data/path_lengths_ship_specific_routes.p', 'rb'))
        self.type_engine_power = pickle.load(open('data/flow_comp_factors_unscaled.p', 'rb'))
        self.type_loaded_speed = pickle.load(open('data/types_loaded_speed.p', 'rb'))
        # self.optimal_flows = opt_flows
        self.optimal_flows = pickle.load(open('data/inputs/non_zero_flows_batch' + str(run) + '.p', 'rb'))

        self.agent_data = {'id': [],
                           'route': [],
                           'combi': [],
                           'generation_hour': [],
                           'time_departed': [],
                           'travel_time': [],
                           'time_in_line': [],
                           'time_charging': [],
                           'battery_size': [],
                           'waited_at': [],
                           'total_charged': []}  # new dict to store data of removed agents, before removing

        self.datacollector = DataCollector(
            model_reporters={"data_completed_trips": "agent_data"},
            agent_reporters={"occupation": (lambda x: get_cs_occupation(x)),
                             "max_occupation": (lambda x: get_max_occupation(x)),
                             "avg_line": (lambda x: get_cs_waiting_line(x)),
                             "max_line": (lambda x: get_max_line_length(x)),

                             # "vessel_status": (lambda x: get_vessel_status(x)),
                             # "vessel_route": (lambda x: get_key(x)),
                             # "combi": (lambda x: get_combi(x)),
                             # "departed_from": lambda x: get_departed_from(x),
                             # "battery_size": (lambda x: get_battery_size(x)),
                             # "battery_level": (lambda x: get_battery_level(x)),
                             # "generated_at": (lambda x: get_generation_time(x)),
                             # "removed_at": (lambda x: get_removal_time(x)),
                             # "station_status": (lambda x: get_station_status(x)),
                             "charging_stations": (lambda x: get_cs(x))
                             })

        self.generate_model()
        self.datacollector.collect(self)

    def generate_model(self):
        """
        generate the simulation model according to the csv file component information

        Warning: the labels are the same as the csv column labels
        """

        y_min, y_max, x_min, x_max = set_lat_lon_bound(
            self.data['Y'].min(),
            self.data['Y'].max(),
            self.data['X'].min(),
            self.data['X'].max(),
            0.05
        )

        # ContinuousSpace from the Mesa package;
        # not to be confused with the SimpleContinuousModule visualization
        self.space = ContinuousSpace(x_max, y_max, True, x_min, y_min)

        for _, row in self.data.iterrows():  # index, row in ...
            # create agents according to model_type
            model_type = row['model_type']
            agent = None
            if model_type == 'link':
                agent = Link(row['name'], self, row['length_m'], row['name'])
                self.links.append(agent.unique_id)
            elif model_type == 'intermediate_node':
                agent = Link(row['name'], self, row['length_m'], row['name'])
                self.intermediate_nodes.append(agent.unique_id)
            elif model_type == 'harbour_with_charging':
                agent = HarbourChargingStation(row['name'], self, row['charging_stations'],
                                               self.charging_station_capacity, row['length_m'], row['name'])
                self.harbours_with_charging.append(agent.unique_id)
            elif model_type == 'harbour':
                agent = Harbour(row['name'], self, row['length_m'], row['name'])
                self.harbours.append(agent.unique_id)
            elif model_type == 'charging_station':
                agent = ChargingStation(row['name'], self, row['charging_stations'], self.charging_station_capacity,
                                        row['length_m'], row['name'])
                self.charging_stations.append(agent.unique_id)
            elif (model_type == 'inserted_node') or (model_type == 'intermediate_node'):
                agent = Intersection(row['name'], self, row['length_m'], row['name'])
                self.inserted_nodes.append(agent.unique_id)

            if agent:
                self.schedule.add(agent)
                y = row['Y']
                x = row['X']
                self.space.place_agent(agent, (x, y))
                agent.pos = (x, y)

    def get_route(self, key):
        # return route
        if key in self.path_ids_dict:
            if random.random() > 0.5:
                return self.path_ids_dict[key]
            # return reversed route
            else:
                return self.path_ids_dict[key][::-1]
        # else raise error, because something is off...
        else:
            print("Error route not found for vessel", self, "travelling (from, to, viaroute):", key)
            self.running = False

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()
        self.datacollector.collect(self)
        # if self.schedule.time == 1:
        #     print("evaluate now:", self.range, self.charging_station_capacity, self.data.charging_stations.max(), self.seed)
        # update hour
        if self.schedule.time % 59:
            if (self.hour + 1) < 24:
                self.hour += 1
            else:
                self.hour = 0

        type_list = list(self.ivs_data.iloc[:, 4:-2])
        df_hour = self.ivs_data.loc[(self.ivs_data.hour == self.hour)]
        df_hour.reset_index(inplace=True, drop=True)

        for i, row in df_hour.iterrows():
            # chance that a vessel is generated is equal trip_count/365
            # (because trip_count is yearly) * (time_step/hours)
            if (row['trip_count'] / 365) * (1 / 60) >= np.random.random():
                # calculate part of route that is captured
                percentage_captured = sum(self.optimal_flows[row['key']]['flows'])
                # now continue with generating only if this vessel should indeed be generated
                if percentage_captured >= np.random.random():
                    # determine vessel type
                    prob = list(df_hour.iloc[i, 4:-2].values / df_hour.iloc[i, 4:-2].sum())
                    to_pick = type_list
                    ship_type = np.random.choice(a=to_pick, size=1, replace=False, p=prob)
                    # print(ship_type, "Vessel departed at", row['origin'], self.hour, ':', self.schedule.time,
                    # "heading to", row['destination'], "via route", row['key'])
                    unique_id = Harbour.vessel_counter  # give unique ID based on Harbour attribute
                    path = self.get_route(row['key'])  # determine path based on route, 50% to depart at port
                    generated_at = path[0]  # store origin
                    generated_by = self.schedule._agents[generated_at]  # store which agent generated this vessel
                    power = self.type_engine_power[ship_type[0]]  # look up engine power based on type
                    speed = (3.6 * 1000 * self.type_loaded_speed[ship_type[0]]) / 60  # from m/s to km/minute

                    battery_size = math.ceil((self.range / (speed * 60)) * power)  # equal r assumed

                    # determine combination that this vessel will use
                    if len(self.optimal_flows[row['key']]['combinations']) == 1:
                        combi = self.optimal_flows[row['key']]['combinations'][0]  # take first if only one option
                    else:
                        pkey = self.optimal_flows[row['key']]['flows']  # else, randomly draw as previously
                        pkey = [i if i > 10e-7 else 0 for i in pkey]
                        pkey = [i / sum(pkey) for i in pkey]

                        pick = np.random.choice(a=np.arange(len(self.optimal_flows[row['key']]['combinations'])),
                                                size=1, replace=False, p=pkey)
                        combi = self.optimal_flows[row['key']]['combinations'][pick.item()]

                    agent = Vessel(unique_id, self, generated_by, path, ship_type[0], battery_size, power, combi,
                                   row['key'], speed)  # instantiate agent and add to vessel
                    self.schedule.add(agent)
                    Harbour.vessel_counter += 1  # make sure next vessel also gets unique id, goes well till 10000

# EOF -----------------------------------------------------------
