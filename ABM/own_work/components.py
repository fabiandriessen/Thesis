from mesa import Agent
from enum import Enum


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

    def step(self):
        pass

    def __str__(self):
        return type(self).__name__ + str(self.unique_id)


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

    def remove(self, vessel):
        self.model.schedule.remove(vessel)
        self.vessel_removed_toggle = not self.vessel_removed_toggle
        print(str(self) + ' REMOVE ' + str(vessel))


# ---------------------------------------------------------------
class ChargingStation(Infra):
    def __init__(self, unique_id, model, charging_stations, length, name):
        super().__init__(unique_id, model)
        self.modules = charging_stations
        self.length = length
        self.name = name
        # capacity is in #vessel/24 hours
        self.charging_speed = 1/((24/self.model.charging_station_capacity)*60*60)
        self.waiting_line = []

    """
    Charges Vessels 
    """


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

    def __init__(self, unique_id, model, charging_stations, length, name):
        super().__init__(unique_id, model)
        self.length = length
        self.name = name
        self.vessel_removed_toggle = False
        self.vessel_counter = 0
        self.generation_frequency = 5
        self.vessel_generated_flag = False
        self.modules = charging_stations
        self.charging_speed = 1/((24/self.model.charging_station_capacity)*60*60)
        self.waiting_line = []

    def step(self):
        # Current plan: vessels generated in step function of the model
        pass

    def remove(self, vessel):
        self.model.schedule.remove(vessel)
        self.vessel_removed_toggle = not self.vessel_removed_toggle
        print(str(self) + ' REMOVE ' + str(vessel))


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

    def __init__(self, unique_id, model, generated_by, path, ship_type, battery_size, combi, location_offset=0):
        super().__init__(unique_id, model)
        # defined attributes
        self.generated_by = generated_by
        self.path_ids = path
        self.ship_type = ship_type
        self.battery_size = battery_size
        self.combi = combi

        # secondary attributes
        self.generated_at_step = model.schedule.steps
        self.location = generated_by
        self.location_offset = location_offset
        self.pos = generated_by.pos
        self.inline = False

        # default values
        self.state = Vessel.State.DRIVE
        self.location_index = 0
        self.waiting_time = 0
        self.waited_at = None
        self.removed_at_step = None
        # 10 km/h translated into meter per min
        self.speed = 10 * 1000 / 60
        # One tick represents 1 minute
        self.step_time = 1

        # departs fully charged if charging station at harbour, otherwise half full
        if self.location.unique_id in combi:
            self.charge = 1
        else:
            self.charge = 0.5

    def __str__(self):
        return "Vessel" + str(self.unique_id) + \
               " +" + str(self.generated_at_step) + " -" + str(self.removed_at_step) + \
               " " + str(self.state) + '(' + str(self.waiting_time) + ') ' + \
               str(self.location) + '(' + str(self.location.vessel_count) + ') ' + str(self.location_offset)

    def step(self):
        """
        Vessel waits or drives at each step
        """
        if self.state == Vessel.State.WAIT:
            if self.inline:
                self.waiting_time = self.get_charging_time(self.location)
            else:
                self.waiting_time = max(self.waiting_time - 1, 0)
                if self.waiting_time == 0:
                    self.waited_at = self.location
                    self.location.waiting_line.remove(self)
                    self.state = Vessel.State.DRIVE

        if self.state == Vessel.State.DRIVE:
            self.drive()

        """
        To print the vessel trajectory at each step
        """
        print(self)

    def drive(self):

        # the distance that vessel drives in a tick
        # speed is global now: can change to instance object when individual speed is needed
        distance = self.speed * self.step_time
        distance_rest = self.location_offset + distance - self.location.length

        if distance_rest > 0:
            # go to the next object
            self.drive_to_next(distance_rest)
        else:
            # remain on the same object
            self.location_offset += distance

    def drive_to_next(self, distance):
        """
        vessel shall move to the next object with the given distance
        """

        self.location_index += 1
        next_id = self.path_ids[self.location_index]
        next_infra = self.model.schedule._agents[next_id]  # Access to protected member _agents

        if (next_id == self.location.unique_id) and (self.location_index != 0):
            # arrive at the sink
            self.arrive_at_next(next_infra, 0)
            self.removed_at_step = self.model.schedule.steps
            self.location.remove(self)
            return

        elif isinstance(next_infra, ChargingStation) or isinstance(next_infra, HarbourChargingStation):
            if next_infra.unique_id in self.combi:
                self.waiting_time = self.get_charging_time(next_infra)
                if self.waiting_time > 0:
                    # arrive at the bridge and wait
                    self.arrive_at_next(next_infra, 0)
                    self.state = Vessel.State.WAIT
                    return
                # else, continue driving

        if next_infra.length > distance:
            # stay on this object:
            self.arrive_at_next(next_infra, distance)
        else:
            # drive to next object:
            self.drive_to_next(distance - next_infra.length)

    def arrive_at_next(self, next_infra, location_offset):
        """
        Arrive at next_infra with the given location_offset
        """
        self.location.vessel_count -= 1
        self.location = next_infra
        self.location_offset = location_offset
        self.location.vessel_count += 1

    def get_charging_time(self, cs):
        # 1 step = 1 min
        if len(cs.waiting_line) < cs.modules:
            charge_needed = self.battery_size * self.charge
            charging_time = charge_needed/cs.charging_speed
            cs.waiting_line.append(self)
            self.inline = False
        else:
            charging_time = 1e6
            self.inline = True

        return charging_time

# EOF -----------------------------------------------------------
