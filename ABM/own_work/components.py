import networkx as nx
from mesa import Agent
from enum import Enum
import math


# ---------------------------------------------------------------
class Infra(Agent):
    """
    Base class for all infrastructure components

    Attributes
    __________
    vessel_count : int
        the number of vessels that are currently in/on (or totally generated/removed by)
        this infrastructure component

    length : float
        the length in meters
    ...

    """

    def __init__(self, unique_id, model, length=0,
                 name='Unknown'):
        super().__init__(unique_id, model)
        self.length = length
        self.name = name
        self.vessel_count = 0
        self.vessel_removed_toggle = False
        self.vessel_generated_flag = False

    def step(self):
        pass

    def __str__(self):
        return type(self).__name__ + str(self.unique_id)

    def remove(self, vessel):
        if self.model.schedule.time >= 60*24:
            vessel.removed_at_step = self.model.schedule.time
            vessel.model.agent_data['id'].append(vessel.unique_id)
            vessel.model.agent_data['route'].append(vessel.route_key)
            vessel.model.agent_data['time_departed'].append(vessel.generated_at_step)
            vessel.model.agent_data['travel_time'].append(vessel.time_driving-1)  # correct for last step
            vessel.model.agent_data['time_in_line'].append(vessel.time_inline)
            vessel.model.agent_data['time_charging'].append(vessel.time_waited)
            vessel.model.agent_data['battery_size'].append(vessel.battery_size)
            vessel.model.agent_data['combi'].append(vessel.combi)
            vessel.model.agent_data['generation_hour'].append(vessel.hour)

        self.model.schedule.remove(vessel)
        self.vessel_removed_toggle = not self.vessel_removed_toggle
        # if vessel.unique_id == 1:
        #     print(vessel)
        # print(str(self) + ' REMOVE ' + str(vessel))


# ---------------------------------------------------------------
class Link(Infra):
    pass


# ---------------------------------------------------------------
class Intersection(Infra):
    pass


# ---------------------------------------------------------------

class Harbour(Infra):
    """
    Harbour generates and removes vessels

    Class Attributes:
    -----------------
    vessel_counter : int
        the number of vessels generated by ALL sources. Used as vessel ID!

    Attributes
    __________
    generation_frequency: int
        the frequency (the number of ticks) by which a vessel is generated

    vessel_generated_flag: bool
        True when a vessel is generated in this tick; False otherwise

    vessel_removed_toggle: bool
        toggles each time when a vessel is removed
    ...

    """
    vessel_counter = 0

    def __init__(self, unique_id, model, length, name):
        super().__init__(unique_id, model)
        self.length = length
        self.name = name
        self.vessel_removed_toggle = False
        self.vessel_generated_flag = False

    def step(self):
        # Current plan: vessels generated in step function of the model
        pass


# ---------------------------------------------------------------
class ChargingStation(Infra):
    def __init__(self, unique_id, model, charging_stations, charging_speed, length, name):
        super().__init__(unique_id, model)
        self.modules = charging_stations
        self.length = length
        self.name = name
        self.vessel_removed_toggle = False
        self.vessel_counter = 0
        self.vessel_generated_flag = False
        self.charging_speed = charging_speed
        self.currently_charging = []
        self.line = []
        self.users = 0
        self.waiters = 0
        self.steps_measuring = 0
        self.max_line_l = 0
        self.max_occupation = 0

    def evaluate_waiting_line(self):
        # only evaluate line if there is a line and if there are charging spots free for the required amount of time
        if self.modules > len(self.currently_charging):
            for i in range(int(self.modules - len(self.currently_charging))):
                if self.line:
                    to_charge = self.line.pop(0)
                    self.currently_charging.append(to_charge)
                    to_charge.inline = False

    def update_usage(self):

        self.steps_measuring += 1
        if self.currently_charging:
            self.users += len(self.currently_charging)
        if self.line:
            self.waiters += len(self.line)

        if self.max_occupation != self.modules:
            self.max_occupation = max(len(self.currently_charging), self.max_occupation)

        if len(self.line) > self.max_line_l:
            self.max_line_l = len(self.line)

    def step(self):
        self.evaluate_waiting_line()
        if self.model.schedule.time >= 60 * 24:
            self.update_usage()


# ---------------------------------------------------------------
class HarbourChargingStation(Infra):
    """
    Creates delay time
    Attributes
    __________
    condition:
        Occupied or Available
    delay_time: int
        the delay (in ticks) caused by this charging station
    ...
    """

    def __init__(self, unique_id, model, charging_stations, charging_speed, length, name):
        super().__init__(unique_id, model)
        self.length = length
        self.name = name
        self.vessel_removed_toggle = False
        self.vessel_counter = 0
        self.vessel_generated_flag = False
        self.modules = charging_stations
        self.charging_speed = charging_speed  # KWh
        self.currently_charging = []
        self.line = []
        self.users = 0
        self.waiters = 0
        self.steps_measuring = 0
        self.max_line_l = 0
        self.max_occupation = 0

    def evaluate_waiting_line(self):
        # only evaluate line if there is a line and if there are charging spots free for the required amount of time
        if self.modules > len(self.currently_charging):
            for i in range(int(self.modules - len(self.currently_charging))):
                if self.line:
                    to_charge = self.line.pop(0)
                    self.currently_charging.append(to_charge)
                    to_charge.inline = False

    def update_usage(self):
        self.steps_measuring += 1
        if self.currently_charging:
            self.users += len(self.currently_charging)
        if self.line:
            self.waiters += len(self.line)

        if self.max_occupation != self.modules:
            self.max_occupation = max(len(self.currently_charging), self.max_occupation)

        if len(self.line) > self.max_line_l:
            self.max_line_l = len(self.line)

    def step(self):
        self.evaluate_waiting_line()
        if self.model.schedule.time >= 60*24:
            self.update_usage()


# ---------------------------------------------------------------
class Vessel(Agent):
    """

    Attributes
    __________
    speed: float
        speed in meter per minute (m/min)

    step_time: int
        the number of minutes (or seconds) a tick represents
        Used as a base to change unites

    state: Enum (DRIVE | WAIT)
        state of the vessel

    location: Infra
        reference to the Infra where the vessel is located

    location_offset: float
        the location offset in meters relative to the starting point of
        the Infra, which has a certain length
        i.e. location_offset < length

    path_ids: Series
        the whole path (origin and destination) where the vessel shall drive
        It consists the Infras' uniques IDs in a sequential order

    location_index: int
        a pointer to the current Infra in "path_ids" (above)
        i.e. the id of self.location is self.path_ids[self.location_index]

    waiting_time: int
        the time the vessel needs to wait

    generated_at_step: int
        the timestamp (number of ticks) that the vessel is generated

    removed_at_step: int
        the timestamp (number of ticks) that the vessel is removed
    ...

    """

    class State(Enum):
        DRIVE = 1
        WAIT = 2

    def __init__(self, unique_id, model, generated_by, path, ship_type, battery_size, power, combi, route_key, speed,
                 location_offset=0):
        super().__init__(unique_id, model)
        # defined attributes
        self.generated_by = generated_by
        self.path_ids = path
        self.ship_type = ship_type
        self.battery_size = battery_size
        self.combi = combi
        self.power = power
        self.route_key = route_key
        self.speed = speed  #type specific

        # secondary attributes
        self.generated_at_step = model.schedule.steps
        self.location = generated_by
        self.location_offset = location_offset
        self.location_index = 0
        self.target_index = 1
        self.pos = generated_by.pos
        self.current_path_length = nx.dijkstra_path_length(self.model.G, self.path_ids[self.location_index],
                                                           self.path_ids[self.location_index + 1], weight='length_m')
        # if self.unique_id == 1:
        #     print('initial path length route', self.route_key, self.current_path_length)
        self.inline = False  # initially False, True if vessel stands inline

        # default values
        self.state = Vessel.State.DRIVE

        # charging related attributes
        self.waiting_time = 0  # variable to keep track of waiting time
        self.time_waited = 0  # variable to keep track the passed waiting time
        self.time_inline = 0  # total time spent waiting at any stations
        self.time_driving = 0  # total time driving
        self.charged_at_dest = 0  # charging time before removed once dest is reached
        self.waited_at = {}  # dict with unique ID where vessel had to wait as a key, and the value is the time spent
        self.remove_if_charged = False  # if True, a vessel should be removed from the model once fully charged

        self.removed_at_step = None
        self.step_time = 1  # One tick represents 1 minute
        self.hour = self.model.hour

        # departs fully charged if charging station at harbour, otherwise half full (consistent with assumption frlm)
        if self.location.unique_id in combi:
            self.charge = 1 * self.battery_size
        else:
            self.charge = 0.5 * self.battery_size

    def __str__(self):
        return "Vessel" + str(self.unique_id) + " +" + str(self.generated_at_step) + " -" + str(self.combi) +\
               " " + str(self.path_ids[self.location_index]) + " " + str(self.state) + '(' + str(self.waiting_time) \
               + ') ' + ' ' + str(self.route_key) + ' ' + str(self.location_offset) + ' ' + str(self.location)

    def step(self):
        """
        Vessel waits or drives at each step
        """
        # update current path length and pos
        # if self.unique_id == 1:
        #     print(self)

        if self.state == Vessel.State.WAIT:
            self.time_waited += 1
            if self.inline:
                self.time_inline += 1
            else:
                self.waiting_time = max(self.waiting_time - 1, 0)
                self.waited_at[self.location.unique_id] += 1
                if self.waiting_time == 0:
                    self.charge = self.battery_size  # battery is fully charged after stop
                    self.location.currently_charging.remove(self)  # no longer charging
                    # now remove if True, else continue journey
                    if self.remove_if_charged:
                        self.removed_at_step = self.model.schedule.steps
                        self.location.remove(self)
                        # print('remove fully charged vessel', self)
                    else:
                        # print('charged vessel continues journey:', self)
                        self.state = Vessel.State.DRIVE
                        self.time_driving += 1

        if self.state == Vessel.State.DRIVE:
            self.time_driving += 1
            self.drive()

        """
        To print the vessel trajectory at each step
        """
        # print(self)

    def drive(self):
        distance = self.speed * self.step_time  # distance covered in tick
        self.charge -= (self.step_time / 60) * self.power
        self.pos = self.update_pos()  # update position
        self.location_offset += distance
        distance_rest = self.location_offset - self.current_path_length  # distance remaining on current path

        if distance_rest > 0:
            if self.path_ids[self.target_index] != self.path_ids[-1]:
                self.location_index += 1
                self.target_index += 1
                self.current_path_length = self.current_path_length + self.update_path_length()

                next_id = self.path_ids[self.location_index]
                next_infra = self.model.schedule._agents[next_id]

                if next_id in self.combi:
                    # charge here
                    self.arrive_at_next(next_infra, distance_rest, correct_overshoot=True)
                    self.get_charging_time(self.location)
                    return
                else:
                    self.arrive_at_next(next_infra, distance_rest)

            elif self.path_ids[self.target_index] == self.path_ids[-1]:
                # arrive at next first (always)
                next_id = self.path_ids[self.target_index]
                next_infra = self.model.schedule._agents[next_id]
                self.arrive_at_next(next_infra, distance_rest, correct_overshoot=True, final_dest=True)
                if next_id in self.combi:
                    self.get_charging_time(self.location)
                    self.charged_at_dest = self.waiting_time
                    self.remove_if_charged = True
                else:
                    self.removed_at_step = self.model.schedule.steps
                    self.location.remove(self)

        if self.charge < 0:
            print("Bug detected, negative charge", self)
            self.model.running = False

    def arrive_at_next(self, next_infra, location_offset, correct_overshoot=False, final_dest=True):
        """
        Arrive at next_infra with the given location_offset
        """
        # print("new path length route", self.route_key, self.current_path_length)
        self.location.vessel_count -= 1
        self.location = next_infra
        if correct_overshoot:
            self.location_offset -= location_offset
            distance = self.speed * self.step_time
            self.charge += (1 - ((distance-location_offset)/distance)) * (self.step_time/60) * self.power
        self.location.vessel_count += 1

        # update location
        if not final_dest:
            self.pos = self.update_pos(location_offset)
        else:
            self.pos = next_infra.pos

    def get_charging_time(self, cs):
        charge_needed = self.battery_size - self.charge  # assumption: always charged 100%
        charging_time = math.ceil((charge_needed / cs.charging_speed) * 60)  # times 60 to convert from hours to minutes
        # check if vessel can charge or joins line
        if cs.line:
            inline = True
            cs.line.append(self)
        else:
            if cs.modules > len(cs.currently_charging):
                inline = False
                cs.currently_charging.append(self)
            else:
                inline = True
                cs.line.append(self)

        # set state, set and store waiting time
        self.state = Vessel.State.WAIT
        self.waiting_time = charging_time
        self.inline = inline
        self.waited_at[cs.unique_id] = 0

    def update_path_length(self):
        new_path = nx.dijkstra_path_length(self.model.G, self.path_ids[self.location_index],
                                           self.path_ids[self.target_index], weight='length_m')
        return new_path

    def update_pos(self, delta_dist=False):

        x1, y1 = self.location.pos
        a = self.location.unique_id
        b = self.path_ids[(self.path_ids.index(a) + 1)]
        x2, y2 = self.model.schedule._agents[b].pos

        current_path_l = self.current_path_length
        if not delta_dist:
            delta_dist = self.speed * self.step_time
        d_x = ((x2 - x1) / current_path_l) * delta_dist
        d_y = ((y2 - y1) / current_path_l) * delta_dist

        new_x = self.pos[0] + d_x
        new_y = self.pos[1] + d_y

        return new_x, new_y

# EOF -----------------------------------------------------------
