from config_SimPy import *
from log_SimPy import *
import environment as env
from config_RL import *
from Def_Scenarios import * 
import time

aseembly_process = "AP2"
# Start timing the computation
start_time = time.time()

demand_uniform = {"Dist_Type": "UNIFORM", "min": 14, "max": 14}

leadtime_uniform = {"Dist_Type": "UNIFORM", "min": 1, "max": 1}
# Define the scenario
create_scenarios()
scenario = {"DEMAND": demand_uniform, "LEADTIME": leadtime_uniform}

# Create environment
simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events = env.create_env(
    I, P, DAILY_EVENTS, aseembly_process)
env.simpy_event_processes(simpy_env, inventoryList, procurementList,
                          productionList, sales, customer, supplierList, daily_events, I, scenario, aseembly_process)

# Print the initial inventory status
if PRINT_SIM:
    print(f"============= Initial Inventory Status =============")
    for inventory in inventoryList:
        print(
            f"{I[aseembly_process][inventory.item_id]['NAME']} Inventory: {inventory.on_hand_inventory} units")

    print(f"============= SimPy Simulation Begins =============")
total_cost = 0
# Run the simulation
for x in range(SIM_TIME):
    print(f"\nDay {(simpy_env.now) // 24+1} Report:")

    # Run the simulation for 24 hours
    simpy_env.run(until=simpy_env.now+24)

    # Print the simulation log every 24 hours (1 day)
    if PRINT_SIM:
        for log in daily_events:
            print(log)
    if PRINT_SIM:
        daily_events.clear()

    env.update_daily_report(inventoryList, aseembly_process)

    cost = env.Cost.update_cost_log(inventoryList)
    # Print the daily cost
    if PRINT_SIM:
        for key in DAILY_COST_REPORT.keys():
            print(f"{key}: {DAILY_COST_REPORT[key]}")
        print(f"Daily Total Cost: {cost}")
    total_cost += cost
    print(f"Cumulative Total Cost: {total_cost}")
    env.Cost.clear_cost()

# Calculate computation time and print it
end_time = time.time()
print(f"\nComputation time (s): {(end_time - start_time):.2f} seconds")
print(f"Computation time (m): {(end_time - start_time)/60:.2f} minutes")