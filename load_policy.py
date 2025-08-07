from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
import envs.simpy_envs.promp_env_nonstationary as simpy_env
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.meta_trainer import Trainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder
from envs.simpy_envs.config_SimPy import *
from envs.simpy_envs.config_folders import *
from envs.simpy_envs.log_SimPy import *
from envs.simpy_envs.scenarios import * 
import numpy as np
import tensorflow as tf
import os
import time
import statistics

def setting_scenario(procurementList, customer, tasks):
    for procurement in procurementList:
        procurement.lead_time = tasks["LEADTIME"]
    customer.demand_qty_dict = tasks["DEMAND"]

def run_simpy(tasks):
    meta_results = []
    for meta_algo in ["ProMP", "MAML"]:
        test_result, actions = simul(tasks, meta_algo)
        meta_results.append(test_result)
    return meta_results, actions

def simul(tasks, meta_algo):
    tf.compat.v1.disable_eager_execution()
    
    config = {
        'seed': 1,
        'baseline': 'LinearFeatureBaseline',
        'env': 'HalfCheetahRandDirecEnv', # Not using this parameter
        'rollouts_per_meta_task': 1,  # number of trajectorys for adapting inner loop
        'max_path_length': SIM_TIME,
        'parallel': True,  # Multi-processing
        'discount': 0.99,
        'gae_lambda': 1,
        'normalize_adv': True,
        'hidden_sizes': (64, 64, 64),
        'learn_std': True,  # whether to learn the standard deviation of the gaussian policy
        'inner_lr': 0.002,  # adaptation step size
        'clip_eps': 0.3,  # clipping range
        'target_inner_step': 0.01,
        'init_inner_kl_penalty': 5e-4,
        'adaptive_inner_kl_penalty': False,  # whether to use an adaptive or fixed KL-penalty coefficient
        'n_itr': 10,  # number of overall training iterations
        'meta_batch_size': 1,  # number of sampled meta-tasks per iterations
        'num_inner_grad_steps': 1,  # number of inner / adaptation gradient steps
    }

    # Set random seed
    set_seed(config['seed'])

    # Create a new TensorFlow graph
    new_graph = tf.Graph()
    with new_graph.as_default():
        # Apply the MetaEnv environment to the graph
        env = simpy_env.MetaEnv(tasks) # apply simpy_env wrapper to env
        sess = tf.compat.v1.Session(graph=new_graph)  # Start a new session with the new graph
        sess.__enter__()

        # Initialize the policy
        policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )

        # Restore model parameters from the saved model
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, os.path.join(f"envs/Saved_Model/{meta_algo}", "model"))  # Restore the model from a checkpoint
        policy.switch_to_pre_update()  # Switch the policy to pre-update mode

        # Initialize result variables
        test_result = {
            "Mean": 0,
            "Variance": 0,
            "Holding cost": 0,
            "Process cost": 0,
            "Delivery cost": 0,
            "Order cost": 0,
            "Shortage cost": 0,
        }

        # Store results for calculating mean and variance later
        mean_data = []
        actions = []
        for test_id in range(NUM_OF_TEST):
            # Update the scenario
            if STATIONARY:
                current_scenario = tasks[test_id]
            else:
                current_scenario = tasks[test_id * 3]

            print(f"Testing with scenario: {current_scenario}")
            obs = env.reset()  # Reset the environment

            total_cost = 0  # Initialize total cost for this test

            # Run the simulation for each day in SIM_TIME
            for day in range(SIM_TIME):
                print(f"\nDay {(env.simpy_env.now) // 24 + 1} Report:")
                if STATIONARY == False:
                    if day == 100:
                        print("Former_Scenario:", current_scenario)
                        current_scenario = tasks[test_id * 3 + 1]
                        setting_scenario(env.procurementList, env.customer, current_scenario)
                        print("After_Scenario:", current_scenario)

                    if day == 150:
                        print("Former_Scenario:", current_scenario)
                        current_scenario = tasks[test_id * 3 + 2]
                        setting_scenario(env.procurementList, env.customer, current_scenario)
                        print("After_Scenario:", current_scenario)

                # Get action from the policy and perform the simulation step
                action, _ = policy.get_action(obs)
                obs, reward, done, cost_dict = env.step(action)
                actions.append(action)
                total_cost -= reward  # Update total cost

            # Update the results based on cost_dict
            for key in cost_dict.keys():
                test_result[key] += cost_dict[key] / NUM_OF_TEST  # Averaging the cost

            mean_data.append(total_cost)
            test_result["Mean"] += float(total_cost) / NUM_OF_TEST  # Averaging the total cost

        # Calculate the variance from the mean data
        test_result["Variance"] = statistics.stdev(mean_data)
        sess.close()
        return test_result, actions
