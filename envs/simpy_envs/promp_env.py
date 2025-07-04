from gym.core import Env
import numpy as np
import random
from envs.simpy_envs.config_SimPy import *
from gym import spaces
import numpy as np
from envs.simpy_envs.config_RL import *
import envs.simpy_envs.environment as env
from envs.simpy_envs.log_SimPy import *
from envs.simpy_envs.log_RL import *
import pandas as pd
import matplotlib.pyplot as plt
from envs.simpy_envs.scenarios import *
class MetaEnv(Env):
    """
    Wrapper around OpenAI gym environments, interface for meta learning
    """
    def __init__(self):
        self.actions = []
        self.all_tasks = create_scenarios()

        self.outer_end = False
        self.mat_count = 1

        if ASSEMBLY_PROCESS == "AP1":
            self.mat_count = 1
        elif ASSEMBLY_PROCESS == "AP2":
            self.mat_count = 3
        elif ASSEMBLY_PROCESS == "AP3":
            self.mat_count = 5
       
        self.assembly_process = ASSEMBLY_PROCESS
        #print("Tensorboard Directory: :", TENSORFLOW_LOGS)
        super(MetaEnv, self).__init__()

        # Scenario initialization for the demand
        if DEMAND_DIST_TYPE == "UNIFORM":
            # demand_dist = {"Dist_Type": "UNIFORM",
            #                "min": 8, "max": 15}
            # demand_dist = {"Dist_Type": "UNIFORM",
            #                "min": 10, "max": 13}
            # demand_dist = {"Dist_Type": "UNIFORM",
            #                "min": 8, "max": 11}
            demand_dist = {"Dist_Type": "UNIFORM",
                           "min": 10, "max": 11}  # Default scenario
        elif DEMAND_DIST_TYPE == "GAUSSIAN":
            demand_dist = {"Dist_Type": "GAUSSIAN",
                           "mean": 11, "std": 4}  # Default scenario
        # Scenario initialization for the demand
        if LEAD_DIST_TYPE == "UNIFORM":
            # leadtime_dist = {"Dist_Type": "UNIFORM",
            #                  "min": 1, "max": 3}
            leadtime_dist = {"Dist_Type": "UNIFORM",
                             "min": 1, "max": 2}  # Default scenario
        elif LEAD_DIST_TYPE == "GAUSSIAN":
            leadtime_dist = {"Dist_Type": "GAUSSIAN",
                             "mean": 3, "std": 3}  # Default scenario
        self.scenario = {"DEMAND": demand_dist, "LEADTIME": leadtime_dist}

        self.shortages = 0
        self.total_reward_over_episode = []
        self.total_reward = 0

        # Record the cumulative value of each cost
        self.cost_dict = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        
        os = []
        # Define action space
        self.action_space = spaces.Box(low = 0, high = 6, shape = (self.mat_count, 1), dtype = int)
        # if self.scenario["Dist_Type"] == "UNIFORM":
        #    k = INVEN_LEVEL_MAX*2+(self.scenario["max"]+1)

        # DAILY_CHANGE + INTRANSIT + REMAINING_DEMAND
        os = spaces.Box(low = 0, high = 41, shape=(len(I[self.assembly_process])*(1+DAILY_CHANGE)+self.mat_count*INTRANSIT+1,1), dtype=int)
        '''
        - Inventory Level of Product
        - Daily Change of Product
        - Inventory Level of WIP
        - Daily Change of WIP
        - Inventory Level of Material
        - Daily Change of Material
        - Demand - Inventory Level of Product
        '''
        self.observation_space = os
        
    def sample_tasks(self, n_tasks):
        """
        Samples task of the meta-environment

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        
        tasks = random.sample(self.all_tasks, n_tasks)
        return tasks

    def set_task(self, task):
        """
        Sets the specified task to the current environment

        Args:
            task: task of the meta-learning environment
        """
        self.scenario = task

    def get_task(self):
        """
        Gets the task that the agent is performing in the current environment

        Returns:
            task: task of the meta-learning environment
        """
        return self.scenario

    def reset(self):
        # Initialize the total reward for the episode
        self.cost_dict = {
            'Holding cost': 0,
            'Process cost': 0,
            'Delivery cost': 0,
            'Order cost': 0,
            'Shortage cost': 0
        }
        # Initialize the simulation environment
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = env.create_env(
            I, P, DAILY_EVENTS, ASSEMBLY_PROCESS)
        env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                                  self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I, self.scenario, ASSEMBLY_PROCESS)
        env.update_daily_report(self.inventoryList, ASSEMBLY_PROCESS)

        state_real = self.get_current_state()
        STATE_DICT.clear()
        DAILY_REPORTS.clear()
        self.actions = []
        return state_real

    def step(self, action):
        self.actions.append(action)
        # Update the action of the agent
        if RL_ALGORITHM == "PPO":
            i = 0
            for _ in range(len(I[self.assembly_process])):
                if I[self.assembly_process][_]["TYPE"] == "Material":
                    # Set action as predicted value
                    if CONSISTENT_ACTION:
                        I[self.assembly_process][_]["LOT_SIZE_ORDER"] = ORDER_QTY[i]
                    else:
                        I[self.assembly_process][_]["LOT_SIZE_ORDER"] =  np.round(action[i])

                    i += 1
        elif RL_ALGORITHM == "DQN":
            pass

        # Capture the current state of the environment
        # current_state = env.cap_current_state(self.inventoryList)
        # Run the simulation for 24 hours (until the next day)
        # Action append

        self.simpy_env.run(until=self.simpy_env.now + 24)
        env.update_daily_report(self.inventoryList, self.assembly_process)
        # Capture the next state of the environment
        state_real = self.get_current_state()
        # Set the next state
        next_state = state_real
        # Calculate the total cost of the day
        cost = env.Cost.update_cost_log(self.inventoryList)
        if PRINT_SIM:
            cost = dict(DAILY_COST_REPORT)
        # Cost Dict update
        for key in DAILY_COST_REPORT.keys():
            self.cost_dict[key] += DAILY_COST_REPORT[key]

        env.Cost.clear_cost()
        reward = -cost
        self.total_reward += reward
        self.shortages += self.sales.num_shortages
        self.sales.num_shortages = 0

        if PRINT_SIM:
            # Print the simulation log every 24 hours (1 day)
            print(f"\nDay {(self.simpy_env.now+1) // 24}:")
            if RL_ALGORITHM == "PPO":
                i = 0
                for _ in range(len(I)):
                    if I[self.assembly_process][_]["TYPE"] == "Raw Material":
                        print(
                            f"[Order Quantity for {I[self.assembly_process][_]['NAME']}] ", action[i])
                        i += 1
            # SimPy simulation print
            for log in self.daily_events:
                print(log)
            print("[Daily Total Cost] ", -reward)
            for _ in cost.keys():
                print(_, cost[_])
            print("Total cost: ", -self.total_reward)
            print("[REAL_STATE for the next round] ",  [
                    item-INVEN_LEVEL_MAX for item in next_state])

        self.daily_events.clear()

        # Check if the simulation is done
        done = self.simpy_env.now >= SIM_TIME * 24  # 예: SIM_TIME일 이후에 종료
        if done == True:
            '''
            if DRL_TENSORBOARD or EXPERIMENT_ADAPTATION:
                self.writer.add_scalar(
                    "reward", self.total_reward, global_step=self.cur_episode)
                # Log each cost ratio at the end of the episode
                for cost_name, cost_value in self.cost_dict.items():
                    self.writer.add_scalar(
                        cost_name, cost_value, global_step=self.cur_episode)
                self.writer.add_scalars(
                    'Cost', self.cost_dict, global_step=self.cur_episode)
                print("Episode: ", self.cur_episode,
                      " / Total reward: ", self.total_reward)

            if self.outer_end == True and self.scenario_batch_size == self.cur_inner_loop:
                self.writer.add_scalar(
                    "inner_end/reward", self.total_reward, global_step=self.cur_episode)
            '''
            #print(self.actions[0])
            self.total_reward_over_episode.append(self.total_reward)
            self.total_reward = 0

        info = {}  # 추가 정보 (필요에 따라 사용)

        return next_state, reward, done, info
    
    def get_current_state(self):
        # Make State for RL
        state = []
        # Update STATE_ACTION_REPORT_REAL
        for id in range(len(I[self.assembly_process])):
            # ID means Item_ID, 7 means to the length of the report for one item
            # append On_Hand_inventory
            state.append(
                STATE_DICT[-1][f"On_Hand_{I[self.assembly_process][id]['NAME']}"]+INVEN_LEVEL_MAX)
            # append changes in inventory
            if DAILY_CHANGE == 1:
                # append changes in inventory
                state.append(
                    STATE_DICT[-1][f"Daily_Change_{I[self.assembly_process][id]['NAME']}"]+INVEN_LEVEL_MAX)
            if INTRANSIT == 1:
                if I[self.assembly_process][id]["TYPE"] == "Material":
                    # append Intransition inventory
                    state.append(
                        STATE_DICT[-1][f"In_Transit_{I[self.assembly_process][id]['NAME']}"])

        # Append remaining demand
        state.append(I[self.assembly_process][0]["DEMAND_QUANTITY"] -
                     self.inventoryList[0].on_hand_inventory+INVEN_LEVEL_MAX)
        return state
    def log_diagnostics(self, paths):
        """
        Logs env-specific diagnostic information

        Args:
            paths (list) : list of all paths collected with this env during this iteration
            prefix (str) : prefix for logger
        """
        print("="*20)
        print("rewards:", sum(paths[0]["rewards"]))
        print("="*20)
        
        return sum(paths[0]["rewards"])
        
 
    