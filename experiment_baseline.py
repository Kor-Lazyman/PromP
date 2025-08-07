from envs.simpy_envs.config_SimPy import *
from envs.simpy_envs.log_SimPy import *
import envs.simpy_envs.environment as env
from envs.simpy_envs.config_RL import *
from envs.simpy_envs.config_folders import * 
from envs.simpy_envs.scenarios import * 
import time
import statistics
import pandas as pd
import load_policy as model
#import umap
import matplotlib.pyplot as plt
# cost check
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
    },
    "AP4": {
        "MAT 1": 1,
        "MAT 2": 2,
        "MAT 3": 1,
        "MAT 4": 1,
        "MAT 5": 1,
        "MAT 6": 1,
        "MAT 7": 1,
        "MAT 8": 2,
        "MAT 9": 1,
        "MAT 10": 1
    }
}

# Start timing the computation
start_time = time.time()
sS_policies = [1, 3, 5]
mean_data = [[], [], []]
holding_test = []

actions = {}
for key in sS_policies:
    actions[f'{key}'] = []
test_result= {
    "Mean": [0, 0, 0],
    "Variance": [0, 0, 0],
    "Holding cost": [0, 0, 0],
    "Process cost": [0, 0, 0],
    "Delivery cost": [0, 0, 0],
    "Order cost": [0, 0, 0],
    "Shortage cost": [0, 0, 0],

}
'''
test_result= {
    "Mean": [0, 0, 0, 0, 0 ,0],
    "Variance": [0, 0, 0, 0, 0 ,0],
    "Holding cost": [0, 0, 0, 0, 0 ,0],
    "Process cost": [0, 0, 0, 0, 0 ,0],
    "Delivery cost": [0, 0, 0, 0, 0 ,0],
    "Order cost": [0, 0, 0, 0, 0 ,0],
    "Shortage cost": [0, 0, 0, 0, 0 ,0],

}
'''
# Setting test_params
scenarios = []
inven_mean_origin = {}
inven_mean_change = {}

for i in range(len(I.keys())):
    #inven_mean_change[f'{i}'] = []
    inven_mean_origin[f'{i}'] = []
if STATIONARY:
    for x in range(NUM_OF_TEST):
        sampled_scenario = random.sample(create_scenarios(), 1)[0]
        scenarios.append(sampled_scenario)
else:
    for x in range(NUM_OF_TEST*3):
        sampled_scenario = random.sample(create_scenarios(), 1)[0]
        scenarios.append(sampled_scenario)

def setting_scenario(procurementList, customer, scenario):
    for procurement in procurementList:
        procurement.lead_time = scenario["LEADTIME"]
    customer.demand_qty_dict = scenario["DEMAND"]

def test_baseline_origin(test_id, policy, policy_id):
     # Update_Scenario
        if STATIONARY:
            current_scenario = scenarios[test_id]

        else:
            current_scenario = scenarios[test_id*3]
        # validation scenario
        '''
        # print(test_id) 
        #current_scenario["DEMAND"] = {"Dist_Type": "UNIFORM", "min": 14, "max": 14} 
        #current_scenario["LEADTIME"] = {"Dist_Type": "UNIFORM", "min": 1, "max": 1}
        # ap1's total_cost = 3208
        '''
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

        total_cost = 0 # temporary total_cost
        holding_test_temp = []
        for day in range(SIM_TIME):
            holding_test_temp.append(inventoryList[0].on_hand_inventory)
            print(f"\nDay {(simpy_env.now) // 24+1} Report:")
            if STATIONARY == False:
                if day == 100:
                    print("Fomer_Scenario:", current_scenario)
                    print("Former_Demand:", customer.demand_qty_dict)
                    print("Former_Leadtime:", procurementList[0].lead_time)
                    current_scenario = scenarios[test_id*3+1]
                    setting_scenario(procurementList, customer, current_scenario)
                    print("After_Scenario:", current_scenario)
                    print("After_Demand:", customer.demand_qty_dict)
                    print("After_Leadtime:", procurementList[0].lead_time)
                if day == 150:
                    print("Fomer_Scenario:", current_scenario)
                    print("Former_Demand:", customer.demand_qty_dict)
                    print("Former_Leadtime:", procurementList[0].lead_time)
                    current_scenario = scenarios[test_id*3+2]
                    setting_scenario(procurementList, customer, current_scenario)
                    print("After_Scenario:", current_scenario)
                    print("After_Demand:", customer.demand_qty_dict)
                    print("After_Leadtime:", procurementList[0].lead_time)
            temp_action = []
            for inventory in inventoryList:
                if I[ASSEMBLY_PROCESS][inventory.item_id]["TYPE"] != "Material":
                    continue

                if SSPOLICY:
                    #if inventoryList[0].on_hand_inventory<I[ASSEMBLY_PROCESS][0]["DEMAND_QUANTITY"]:
                        # print(I[ASSEMBLY_PROCESS][inventory.item_id]["NAME"]) # Check item_type
                    if inventory.on_hand_inventory + inventory.in_transition_inventory <= policy: # 밖에서는 0시의 inventory를 감지 할 수 없어서 material이 배송 되는 순간의 inven이 없다고 간주
                            I[ASSEMBLY_PROCESS][inventory.item_id]["LOT_SIZE_ORDER"] = I[ASSEMBLY_PROCESS][0]["DEMAND_QUANTITY"]*Maximum_daily_consumption[ASSEMBLY_PROCESS][I[ASSEMBLY_PROCESS][inventory.item_id]["NAME"]]
                    else:
                        I[ASSEMBLY_PROCESS][inventory.item_id]["LOT_SIZE_ORDER"] = 0
                    temp_action.append(I[ASSEMBLY_PROCESS][inventory.item_id]["LOT_SIZE_ORDER"])
            actions[f'{policy}'].append(temp_action)
            # Run the simulation for 24 hours
            simpy_env.run(until=simpy_env.now+24)
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
                test_result[key][policy_id] += DAILY_COST_REPORT[key]/NUM_OF_TEST
            
            env.Cost.clear_cost()
        holding_test.append(sum(holding_test_temp)/200)
        mean_data[policy_id].append(total_cost)
        test_result["Mean"][policy_id] += total_cost/NUM_OF_TEST
# print(len(scenarios)) # validation all scenarios


def main():
    # Run the simulation
    for test_id in range(NUM_OF_TEST):
        for policy_id in range(len(sS_policies)):
           #test_baseline(test_id, policy)
           test_baseline_origin(test_id, sS_policies[policy_id], policy_id)

    for test_id in range(NUM_OF_TEST):
            test_result["Variance"][policy_id] = statistics.stdev(mean_data[policy_id])
    
    #시나리오 변경에 따른 테스트 변경 필요(nonstationary에선 s1,s2,s3합하셈)
    meta_results, actions_model = model.run_simpy(scenarios)
    for meta_result in meta_results:
        for key in test_result.keys():
            test_result[key].append(meta_result[key])
    
    df = pd.DataFrame(test_result)
    print(CSV_LOG)
    df.to_csv(os.path.join(CSV_LOG, "Test_Result.csv"))
    '''
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean")
    umap_model_2 = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine")
    X_umap_1 = umap_model.fit_transform(actions['1'])
    X_umap_2 = umap_model.fit_transform(actions['3'])
    X_umap_3 = umap_model.fit_transform(actions['5'])
    X_umap_model_1 = umap_model.fit_transform(actions_model)
    X_umap_4 = umap_model_2.fit_transform(actions['1'])
    X_umap_5 = umap_model_2.fit_transform(actions['3'])
    X_umap_6 = umap_model_2.fit_transform(actions['5'])
    X_umap_model_2 = umap_model.fit_transform(actions_model)
    # print(actions_model)
    #print(len(actions_model))
    # 결과 시각화
    # X_umap_1, X_umap_2, X_umap_3를 각각 다른 색으로 시각화
    X = np.random.rand(200, 5)
    plt.scatter(X_umap_1[:, 0], X_umap_1[:, 1], c='r', label='Actions pollicy 1', alpha=0.5)
    plt.scatter(X_umap_2[:, 0], X_umap_2[:, 1], c='g', label='Actions pollicy 3', alpha=0.5)
    plt.scatter(X_umap_3[:, 0], X_umap_3[:, 1], c='b', label='Actions pollicy 5', alpha=0.5)
    plt.scatter(X_umap_model_1[:, 0], X_umap_model_1[:, 1], c='y', label='Actions model', alpha=0.5)
    plt.legend()
    plt.title("UMAP projection")
    plt.show()

    plt.scatter(X_umap_4[:, 0], X_umap_4[:, 1], c='r', label='Actions pollicy 1', alpha=0.5)
    plt.scatter(X_umap_5[:, 0], X_umap_5[:, 1], c='g', label='Actions pollicy 3', alpha=0.5)
    plt.scatter(X_umap_6[:, 0], X_umap_6[:, 1], c='b', label='Actions pollicy 5', alpha=0.5)
    plt.scatter(X_umap_model_2[:, 0], X_umap_model_2[:, 1], c='y', label='Actions model', alpha=0.5)
    plt.legend()
    plt.title("UMAP projection")
    plt.show()
    '''
main()

