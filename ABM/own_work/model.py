from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Harbour, HarbourChargingStation, ChargingStation, Link, Intersection, Vessel
import pandas as pd
from collections import defaultdict
import networkx as nx
import pickle
import numpy as np
import random

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

    def __init__(self, seed=None, x_max=500, y_max=500, x_min=0, y_min=0):
        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = pickle.load(open('data/paths.p', 'rb'))
        self.space = None
        self.G = pickle.load(open('data/network.p', 'rb'))
        self.ivs_data = pickle.load(open('data/df_random.p', 'rb'))
        self.data = pd.read_csv('data/df_abm.csv')
        self.intermediate_nodes = []
        self.harbours = []
        self.harbours_with_charging = []
        self.links = []
        self.charging_stations = []
        self.inserted_nodes = []
        self.hour = 0
        self.charging_station_capacity = 5
        self.battery_sizes = pickle.load(open('data/flow_comp_factors.p', 'rb'))
        self.feasible_combinations = pickle.load(open('data/feasible_comb.p', 'rb'))

        self.generate_model()

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
                agent = HarbourChargingStation(row['name'], self, row['charging_stations'], row['length_m'], row['name'])
                self.harbours_with_charging.append(agent.unique_id)
            elif model_type == 'harbour':
                agent = Harbour(row['name'], self, row['length_m'], row['name'])
                self.harbours.append(agent.unique_id)
            elif model_type == 'charging_station':
                agent = ChargingStation(row['name'], self, row['charging_stations'], row['length_m'], row['name'])
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

    def get_route(self, harbour, key):
        # return route
        if (harbour == key[0]) and (key in self.path_ids_dict):
            return self.path_ids_dict[key] + self.path_ids_dict[key][::-1]
        # return reversed route
        elif (harbour == key[1]) and (key in self.path_ids_dict):
            return self.path_ids_dict[key][::-1] + self.path_ids_dict[key]
        # else raise error, because something is off...
        else:
            print("Error route not found for vessel", self, "travelling (from, to, viaroute):", key)
            self.running = False

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()

        # update hour
        if self.schedule.time % 59:
            if (self.hour + 1) < 24:
                self.hour += 1
            else:
                self.hour = 0

        type_list = list(self.ivs_data.iloc[:, 4:-2])
        df_1 = self.ivs_data.loc[(self.ivs_data.hour == self.hour)]
        harbours = list(set(list(df_1.origin.unique()) + list(df_1.destination.unique())))

        # for each harbour
        for harbour in harbours:
            a = df_1.loc[(df_1.origin == harbour) | (df_1.destination == harbour)]
            # for each origin destination pair that this harbour is part of
            for i, j in enumerate(a.index):
                # chance that a vessel is generated is equal to 1/2 (either spawn at origin or dest) * trip_count/365
                # (because trip_count is yearly) *(time_step/hours)
                # pickle.dump(self.schedule._agents, open('data/schedule.p', 'wb'))
                if 0.5 * (df_1.trip_count[j] / 365) * (1 / 60) >= np.random.random():
                    # determine vessel type
                    prob = list(a.iloc[i, 4:-2].values / a.iloc[i, 4:-2].sum())
                    to_pick = type_list
                    ship_type = np.random.choice(a=to_pick, size=1, replace=False, p=prob)
                    print(ship_type, "Vessel departed at", df_1.origin[j], self.hour, ':', self.schedule.time,
                          "heading to", df_1.destination[j], "via route", df_1.key[j])
                    unique_id = Harbour.vessel_counter
                    path = self.get_route(harbour, df_1.key[j])
                    generated_at = path[0]
                    pickle.dump(self.schedule._agents, open('data/schedule.p', 'wb'))
                    generated_by = self.schedule._agents[generated_at]
                    battery_size = self.battery_sizes[ship_type[0]]
                    print(df_1.key[j])
                    print(self.feasible_combinations[df_1.key[j]])
                    combi = random.choice(self.feasible_combinations[df_1.key[j]])
                    agent = Vessel(unique_id, self, generated_by, path, ship_type, battery_size, combi)
                    self.schedule.add(agent)
                    Harbour.vessel_counter += 1

# EOF -----------------------------------------------------------
