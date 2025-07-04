from envs.simpy_envs.config_SimPy import *
from envs.simpy_envs.config_RL import *
import itertools
import random
import numpy as np
from collections import defaultdict

# 1. Task Sampling Related

def create_scenarios():
    """
    Creates a set of tasks (e.g., environment or scenario definitions).
    
    This function defines the demand and lead time ranges and creates all possible 
    combinations of these parameters for the tasks in the environment.
    
    Returns:
        scenarios (list): A list of dictionaries representing all possible scenarios 
                          combining demand and leadtime configurations.
    """
    # DEMAND
    demand_uniform_range = [
        (i, j)
        for i in range(10, 14)  # Range for demand min values
        for j in range(i, 14)  # Range for demand max values
        if i <= j
    ]
    
    # Define the uniform demand distribution
    demand_uniform = [
        {"Dist_Type": "UNIFORM", "min": 14, "max": 14}
        for min_val, max_val in demand_uniform_range
    ]

    # LEADTIME
    leadtime_uniform_range = [
        (i, j)
        for i in range(1, 4)  # Range for lead time min values
        for j in range(i, 4)  # Range for lead time max values
        if i <= j
    ]
    
    # Define the uniform lead time distribution
    leadtime_uniform = [
        {"Dist_Type": "UNIFORM", "min": 1, "max": 1}
        for min_val, max_val in leadtime_uniform_range
    ]

    # Create all combinations of demand and leadtime
    scenarios = list(itertools.product(demand_uniform, leadtime_uniform))
    scenarios = [{"DEMAND": demand, "LEADTIME": leadtime} for demand, leadtime in scenarios]

    return scenarios

def sample_tasks(all_tasks, num_tasks=NUM_TASKS):
    """
    Randomly samples a subset of tasks from the entire set of tasks.
    
    Args:
        all_tasks (list): List of all available tasks.
        num_tasks (int): The number of tasks to sample.

    Returns:
        tasks (list): A list containing the sampled tasks.
    """
    tasks = random.sample(all_tasks, num_tasks)
    return tasks

# 2. Trajectory Sampling

def sample_trajectories(agent, env, device):
    """
    Collects a trajectory from the given policy and environment.
    
    This function selects actions for MultiDiscrete environments and stores the trajectory.
    
    Args:
        agent (nn.Module): The agent that makes decisions.
        env (gym.Env): The environment used for simulation.
        device (torch.device): The device (CPU/GPU) to run the model on.

    Returns:
        sampled_trajectories (list): A list of tuples containing state, action, reward, 
                                      next_state, done flag, and log probability.
    """
    state = env.reset()
    done = False
    sampled_trajectories = []

    # Collect trajectory τ = {(st, at, rt, st+1, dt)}
    for day in range(SIM_TIME):
        action, log_prob = select_action(state, agent, device)
        next_state, reward, done, info = env.step(action)
        sampled_trajectories.append((state, action, reward, next_state, done, log_prob))
        state = next_state
    
    return sampled_trajectories

def collect_trajectories(states, env, agent, device):
    """
    Collects a trajectory for each state in the given states list.
    
    Args:
        states (list): List of states from which trajectories should be collected.
        env (gym.Env): The environment used for simulation.
        agent (nn.Module): The agent that makes decisions.
        device (torch.device): The device (CPU/GPU) to run the model on.

    Returns:
        trajectories (list): A list of tuples containing state, action, reward, next_state, 
                              done flag, and log probability for each trajectory.
    """
    trajectories = []
    for state in states:
        action, log_prob = select_action(state, agent, device)
        next_state, reward, done, info = env.step(action)
        trajectories.append((state, action, reward, next_state, done, log_prob))
        state = next_state
    
    return trajectories

def make_trajectorie_batches(trajectories, batch_size):
    """
    Splits the trajectory data into batches of a specified size.
    
    Args:
        trajectories (list): A list of trajectories to be split.
        batch_size (int): The size of each batch.

    Returns:
        sampled_trajectories (list): A list of batches, where each batch is a sublist of trajectories.
    """
    dataset_size = SIM_TIME
    indices = np.arange(dataset_size)
    
    np.random.shuffle(indices)
    sampled_trajectories = []

    for i in range(0, dataset_size, batch_size):
        sampled_trajectories.append(trajectories[i : i + batch_size])

    return sampled_trajectories
