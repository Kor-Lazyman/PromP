from envs.simpy_envs.config_SimPy import *
from envs.simpy_envs.log_SimPy import *
import envs.simpy_envs.environment as env
from envs.simpy_envs.config_RL import *
from envs.simpy_envs.config_folders import * 
from envs.simpy_envs.scenarios import * 
import time
import statistics
import pandas as pd
# cost check
test_result= {
    "Mean": [0, 0, 0],
    "Variance": [0, 0, 0],
    "Holding cost": [0, 0, 0],
    "Process cost": [0, 0, 0],
    "Delivery cost": [0, 0, 0],
    "Order cost": [0, 0, 0],
    "Shortage cost": [0, 0, 0],


}

# Setting test_params
scenarios = None
NUM_OF_TEST = 20
STATIONARY = True

if STATIONARY:
    scenarios = random.sample(create_scenarios(), NUM_OF_TEST)
else:
    scenarios = random.sample(create_scenarios(), NUM_OF_TEST*3)

def setting_scenario(procurementList, customer, scenarios):
    for procurement in procurementList:
        procurement.lead_time_dict = scenarios["LEADTIME"]
    customer.scenario = scenarios["DEMAND"]


# print(len(scenarios)) # validation all scenarios
Maximum_daily_consumption ={
    "AP1":{
        "MAT 1": 2
    },
    "AP2": {
        "MAT 1": 2,
        "MAT 2": 4,
        "MAT 3": 2 
    },
    "AP3": {
        "MAT 1": 2,
        "MAT 2": 4,
        "MAT 3": 2,
        "MAT 4": 2,
        "MAT 5": 2
    }
}
# Start timing the computation
start_time = time.time()
sS_policys = [1, 3, 5]
mean_data = [[], [], []]
def main():
    # Run the simulation
    for test_id in range(20):
        for policy in sS_policys:
            # Update_Scenario
            if STATIONARY:
                current_scenario = scenarios[test_id]

            else:
                current_scenario = scenarios[test_id*3]
            #print(test_id)  # validation scenario
            # Create environment
            simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events = env.create_env(
                I, P, DAILY_EVENTS)
            setting_scenario(procurementList, customer, current_scenario)
            env.simpy_event_processes(simpy_env, inventoryList, procurementList,
                                    productionList, sales, customer, supplierList, daily_events, I, current_scenario)
            # Print the initial inventory status
            if PRINT_SIM:
                print(f"============= Initial Inventory Status =============")
                for inventory in inventoryList:
                    print(
                        f"{I[ASSEMBLY_PROCESS][inventory.item_id]['NAME']} Inventory: {inventory.on_hand_inventory} units")

                print(f"============= SimPy Simulation Begins =============")
            total_cost = 0
            for pro in procurementList:
                pro.policy = policy

            for day in range(SIM_TIME):
                print(f"\nDay {(simpy_env.now) // 24+1} Report:")
                if STATIONARY == False:
                    if day == 100:
                        # print(procurementList[0].lead_time_dict) # validation change scenarios
                        current_scenario = scenarios[test_id*3+1]
                        setting_scenario(procurementList, customer, current_scenario)
                        # print(procurementList[0].lead_time_dict)
                    if day == 150:
                        # print(procurementList[0].lead_time_dict)
                        current_scenario = scenarios[test_id*3+2]
                        setting_scenario(procurementList, customer, current_scenario)
                        # print(procurementList[0].lead_time_dict)
                # Run the simulation for 24 hours
                simpy_env.run(until=simpy_env.now+24)
                '''
                # for validation Demand
                print("DEMAND", I[ASSEMBLY_PROCESS][0]["DEMAND_QUANTITY"])
                '''
                # Print the simulation log every 24 hours (1 day)
                if PRINT_SIM:
                    for log in daily_events:
                        print(log)

                if PRINT_SIM:
                    daily_events.clear()

                env.update_daily_report(inventoryList)
                cost = env.Cost.update_cost_log(inventoryList)

                total_cost += cost
                print(f"Cumulative Total Cost: {total_cost}")
                for key in DAILY_COST_REPORT.keys():
                    test_result[key][policy//2] += DAILY_COST_REPORT[key]/20
                
                env.Cost.clear_cost()
            mean_data[policy//2].append(total_cost)
            test_result["Mean"][policy//2] += total_cost/20
        
        # Calculate computation time and print it
        end_time = time.time()
        print(f"\nComputation time (s): {(end_time - start_time):.2f} seconds")
        print(f"Computation time (m): {(end_time - start_time)/60:.2f} minutes")

    for x in range(3):
        test_result["Variance"][x] = statistics.stdev(mean_data[x])

    df = pd.DataFrame(test_result)
    print(CSV_LOG)
    df.to_csv(os.path.join(CSV_LOG, "Test_Result.csv"))
main()