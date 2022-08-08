from mesa.visualization.ModularVisualization import ModularServer
from ContinuousSpace.SimpleContinuousModule import SimpleCanvas
from model import VesselElectrification
from components import Intersection, Harbour, ChargingStation, HarbourChargingStation, Link, Infra, Vessel

"""
Run simulation with Visualization 
Print output at terminal
"""


# ---------------------------------------------------------------
def agent_portrayal(agent):
    """
    Define the animation methode

    Only circles and rectangles are possible
    Both can be labelled
    """

    # define shapes
    portrayal = {
        "Shape": "circle",  # rect | circle
        "Filled": "true",
        "Color": "Khaki",
        "r": 2,
        # "w": max(agent.population / 100000 * 4, 4),  # for "Shape": "rect"
        # "h": max(agent.population / 100000 * 4, 4)
    }

    if (isinstance(agent, Harbour)) or (isinstance(agent, HarbourChargingStation)):
        if agent.vessel_generated_flag:
            portrayal["Color"] = "green"
        else:
            portrayal["Color"] = "red"

        if agent.vessel_removed_toggle:
            portrayal["Color"] = "LightSkyBlue"
        else:
            portrayal["Color"] = "LightPink"

    elif isinstance(agent, Link):
        portrayal["Color"] = "Tan"

    elif isinstance(agent, Intersection):
        portrayal["Color"] = "DeepPink"

    elif isinstance(agent, ChargingStation):
        portrayal["Color"] = "dodgerblue"

    elif isinstance(agent, Vessel):
        portrayal['Color'] = "orange"

    if isinstance(agent, (Harbour, HarbourChargingStation)):
        portrayal["r"] = max(agent.currently_charging+len(agent) * 4, 2)
    # elif isinstance(agent, Infra):
    #     portrayal["r"] = max(agent.vessel_count * 4, 2)

    # define text labels
    # if isinstance(agent, Infra) and agent.name != "":
    #     portrayal["Text"] = agent.name
    #     portrayal["Text_color"] = "DarkSlateGray"

    return portrayal


# ---------------------------------------------------------------
"""
Launch the animation server 
Open a browser tab 
"""

canvas_width = 1000
canvas_height = 1000

space = SimpleCanvas(agent_portrayal, canvas_width, canvas_height)

server = ModularServer(VesselElectrification,
                       [space],
                       "Transport Model Demo",
                       {"seed": 1234567})

# The default port
server.port = 8521
server.launch()
