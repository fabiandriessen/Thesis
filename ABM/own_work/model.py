from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Harbour, HarbourChargingStation, ChargingStation, Link, Intersection
import pandas as pd
from collections import defaultdict
import networkx as nx
import pickle
import numpy as np

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

    # file_name = '../data/demo-4.csv'
    file_name = 'data/df_abm.csv'

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
            print(row['id'])
            if model_type == 'link':
                agent = Link(row['id'], self, row['length_m'], row['name'])
                self.links.append(agent.unique_id)
            elif model_type == 'intermediate_node':
                agent = Link(row['id'], self, row['length_m'], row['name'])
                self.intermediate_nodes.append(agent.unique_id)
            elif model_type == 'harbour_with_charging':
                agent = HarbourChargingStation(row['id'], self, row['charging_stations'], row['length_m'], row['name'])
                self.harbours_with_charging.append(agent.unique_id)
            elif model_type == 'harbour':
                agent = Harbour(row['id'], self, row['length_m'], row['name'])
                self.harbours.append(agent.unique_id)
            elif model_type == 'charging_station':
                agent = ChargingStation(row['id'], self, row['charging_stations'], row['length_m'], row['name'])
                self.charging_stations.append(agent.unique_id)
            elif (model_type == 'inserted_node') or (model_type == 'intermediate_node'):
                agent = Intersection(row['id'], self, row['length_m'], row['name'])
                self.inserted_nodes.append(agent.unique_id)

            if agent:
                self.schedule.add(agent)
                y = row['Y']
                x = row['X']
                self.space.place_agent(agent, (x, y))
                agent.pos = (x, y)

    def get_route(self, origin, destination, route_v):
        # return route
        if (origin, destination, route_v) in self.path_ids_dict:
            return self.path_ids_dict[origin, destination, route_v]
        # return reversed route
        elif (destination, origin, route_v) in self.path_ids_dict:
            return self.path_ids_dict[destination, origin, route_v][::-1]
        # else raise error, because something is off...
        else:
            print("Error route not found for vessel", self, "travelling (from, to, viaroute):", origin, destination,
                  route_v)
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

        for harbour in list(set(list(df_1.origin.unique()) + list(df_1.destination.unique()))):
            a = df_1.loc[(df_1.origin == harbour) | (df_1.destination == harbour)]
            for i, j in enumerate(a.index):
                # chance that a vessel is generated is equal to 1/2 (either spawn at origin or dest) * trip_count/365
                # (because trip_count is yearly) *(time_step/hours)
                if 0.5 * (df_1.trip_count[j] / 365) * (1 / 60) >= np.random.random():
                    # determine vessel type
                    prob = list(a.iloc[i, 4:-2].values / a.iloc[i, 4:-2].sum())
                    to_pick = type_list
                    ship_type = np.random.choice(a=to_pick, size=1, replace=True, p=prob)
                    print(ship_type, "Vessel departed at", df_1.origin[j], self.hour, ':', self.schedule.time,
                          "heading to", df_1.destination[j], "via route", df_1.key[j])
                    # now this vessel must be placed at origin/dest

# EOF -----------------------------------------------------------
