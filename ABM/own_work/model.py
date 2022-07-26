from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Source, Sink, Harbour, ChargingStation, HarbourChargingStation, Link, Intersection
import pandas as pd
from collections import defaultdict
import networkx as nx
import pickle


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
        self.data = pd.read_csv('data/df_abm.csv')
        self.harbours = list(self.data.loc[(self.data.model_type == 'harbour') |
                                           (self.data.model_type == 'harbour_with_charging')].node_id)
        self.links = []
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

        for df in self.data:
            for _, row in df.iterrows():  # index, row in ...
                # create agents according to model_type

                model_type = row['model_type'].strip()
                agent = None

                name = row['name']
                if pd.isna(name):
                    name = ""
                    print("error, nameless entry")
                else:
                    name = name.strip()

                if model_type == 'link':
                    agent = Link(row['id'], self, row['length_m'], name)
                    self.links.append(agent.unique_id)
                elif model_type == 'harbour_with_charging':
                    agent = HarbourChargingStation(row['id'], self, name)
                elif model_type == 'harbour':
                    agent = Harbour(row['id'], self, row['length'], name)
                elif model_type == 'charging_station':
                    agent = ChargingStation(row['id'], self, row['length'], name)
                elif (model_type == 'inserted_node') or (model_type == 'intermediate_node'):
                    agent = Intersection(row['id'], self, row['length'], name)

                if agent:
                    self.schedule.add(agent)
                    y = row['Y']
                    x = row['X']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)

    def get_straight_route(self, source):
        return self.get_route(source, None)

    def get_random_route(self, source):
        """
        pick up a random route given an origin
        """
        while True:
            # different source and sink
            sink = self.random.choice(self.sinks)
            if sink is not source:
                break
        return self.get_route(source, sink)

    def get_route(self, source, sink):
        if (source, sink) in self.path_ids_dict:
            return self.path_ids_dict[source, sink]
        else:
            path_ids = pd.Series(nx.shortest_path(self.G, source, sink))
            self.path_ids_dict[source, sink] = path_ids
            return path_ids

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()

# EOF -----------------------------------------------------------
