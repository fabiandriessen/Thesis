from model import VesselElectrification
import pickle
"""
    Run simulation
    Print output at terminal
"""

# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick 1 minute
run_length = 6 * 24 * 60

# run time 1000 ticks
# run_length = 1000

seed = 1234567

sim_model = VesselElectrification(seed=seed)

# Check if the seed is set
print("SEED " + str(sim_model._seed))

# One run with given steps
for i in range(run_length):
    sim_model.step()

agent_data = sim_model.datacollector.get_agent_vars_dataframe()
model_data = sim_model.datacollector.get_model_vars_dataframe()

pickle.dump(agent_data, open('data/agent_data.p', "wb"))
pickle.dump(model_data, open('data/model_data.p', "wb"))
