import os
import shutil
from config_SimPy import *

# Using correction option
USE_CORRECTION = False

BUFFER_SIZE = 100000
BATCH_SIZE = 20  
LEARNING_RATE = 3e-4
GAMMA = 0.99
CLIP_EPSILON = 0.2
UPDATE_STEPS = 10  
GAE_LAMBDA = 0.95
ENT_COEF = 0.0
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
N_MULTIPROCESS = 5
# def Create_scenario():
#     if DEMAND_DIST_TYPE == "UNIFORM":
#         # Uniform distribution
#         # param_min = random.randint(9, 14)
#         # param_max = random.randint(param_min, 14)
#         param_min = random.randint(10, 13)
#         param_max = random.randint(param_min, 13)
#         demand_dist = {"Dist_Type": DEMAND_DIST_TYPE,
#                        "min": param_min, "max": param_max}
#     elif DEMAND_DIST_TYPE == "GAUSSIAN":
#         # Gaussian distribution
#         param_mean = random.randint(9, 13)
#         param_std = random.randint(1, 4)
#         demand_dist = {"Dist_Type": DEMAND_DIST_TYPE,
#                        "mean": param_mean, "std": param_std}

#     if LEAD_DIST_TYPE == "UNIFORM":
#         # Uniform distribution
#         param_min = random.randint(1, 3)
#         param_max = random.randint(param_min, 3)
#         leadtime_dist = {"Dist_Type": LEAD_DIST_TYPE,
#                          "min": param_min, "max": param_max}
#     elif LEAD_DIST_TYPE == "GAUSSIAN":
#         # Gaussian distribution
#         # Lead time의 최대 값은 Action Space의 최대 값과 곱하였을 때 INVEN_LEVEL_MAX의 2배를 넘지 못하게 설정 해야 함 (INTRANSIT이 OVER되는 현상을 방지 하기 위해서)
#         param_mean = random.randint(1, 3)
#         param_std = random.randint(1, 3)
#         leadtime_dist = {"Dist_Type": LEAD_DIST_TYPE,
#                          "mean": param_mean, "std": param_std}
#     scenario = {"DEMAND": demand_dist, "LEADTIME": leadtime_dist}

#     return scenario

# Define dir's path
DRL_TENSORBOARD = False  # When True for DRL
EXPERIMENT_MAML = True  # When True for EXPERIMENT_MAML
EXPERIMENT_ADAPTATION = False  # When True for EXPERIMENT_ADAPTATION
GRAPH_WRITER = True
# Episode
N_EPISODES = 3000  # Default: 5000

# RL algorithms
RL_ALGORITHM = "PPO"  # "DP", "DQN", "DDPG", "PPO", "SAC"
# Assembly Process 3
# BEST_PARAMS = {'LEARNING_RATE': 0.0006695881981942652,
#                'GAMMA': 0.917834573740, 'BATCH_SIZE': 8, 'N_STEPS': 600}

# Lead time의 최대 값은 Action Space의 최대 값과 곱하였을 때 INVEN_LEVEL_MAX의 2배를 넘지 못하게 설정 해야 함 (INTRANSIT이 OVER되는 현상을 방지 하기 위해서)
ACTION_SPACE = [0, 1, 2, 3, 4, 5]

# Action 값 고정 여부 -> config_SimPy.py에서 고정값 설정
CONSISTENT_ACTION = False  # True: Action 값 고정 / False: RL 에이전트에 따라 Action 값 변동

# Hyperparameter optimization
OPTIMIZE_HYPERPARAMETERS = False
N_TRIALS = 20  # 50

# Evaluation
N_EVAL_EPISODES = 5  # 100

# Export files
DAILY_REPORT_EXPORT = False
STATE_TRAIN_EXPORT = False
STATE_TEST_EXPORT = False

# Visualize_Graph
VIZ_INVEN_LINE = False
VIZ_INVEN_PIE = False
VIZ_COST_PIE = False
VIZ_COST_BOX = False



# Non-stationary demand
mean_demand = 100
standard_deviation_demand = 20


# tensorboard --logdir="~\tensorboard_log"
# http://localhost:6006/
