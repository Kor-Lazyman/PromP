from ProMP_inner import *
from ProMP_optimizer import OuterLoopOptimizer
from config_RL import *
from config_SimPy import *
from config_folders import *
from GymWrapper import *
from Def_Scenarios import *
from model import *
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import random
import time
import torch
import os

def train(theta):
    writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
    env = GymInterface()
    beta = BETA
    action_dim = [len(ACTION_SPACE) for _ in range(env.mat_count)]
    state_dim = len(env.reset())
    batch_size = 16
    convergence_threshold = 1e-4
    scenarios = create_scenarios()
    task_batch, test_batch = split_scenarios(scenarios)
    start_time = time.time()

    # Step 3: tasks 샘플링
    sample_task_batch = random.sample(task_batch, 5)
    sample_test_batch = random.sample(test_batch, 5)

    # --- Inner/Outer 루프 프로파일링 ---
    with record_function("meta_iteration"):
            inner_optim = []
            post_updated_trajectories = []

            for n in range(META_PARAMETER_UPDATE):
                meta_gradients = []
                for task_idx, task in enumerate(sample_task_batch):
                    env.reset()
                    env.scenario = task

                    if n == 0:
                        with record_function("inner_first_optimization"):
                            inner = inner_loop(
                                state_dim, action_dim, theta, env, 1, ALPHA
                            )
                            theta_old, post_traj = inner._first_optimization()
                            inner_optim.append(OuterLoopOptimizer(ALPHA, theta_old))
                            post_updated_trajectories.append(post_traj)

                with record_function("compute_meta_gradient"):
                    grad = inner_optim[task_idx].optimizer(
                        post_updated_trajectories[task_idx],
                        theta, state_dim, action_dim
                    )
                    meta_gradients.append(grad)

            # meta-update
            with record_function("meta_update"):
                total_gradients = [
                    beta * sum(layer_grads)
                    for layer_grads in zip(*meta_gradients)
                ]
                for param, grad in zip(theta.parameters(), total_gradients):
                    param.data.add_(grad)
            # 테스트 평가
            total_reward = 0.0
            for task in sample_test_batch[:1]:
                env.reset()
                env.scenario = task
                test_model = inner_loop(state_dim, action_dim, theta, env, 1, ALPHA)
                _, reward = test_model.simulation(theta)
                total_reward += reward
            print(f"Iteration time: {time.time() - start_time:.2f}s")
if __name__ == '__main__':
    writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
    env = GymInterface()
    beta = BETA
    action_dim = [len(ACTION_SPACE) for _ in range(env.mat_count)]
    state_dim = len(env.reset())
    batch_size = 16
    convergence_threshold = 1e-4
    scenarios = create_scenarios()
    task_batch, test_batch = split_scenarios(scenarios)
    theta = ActorCritic(state_dim, action_dim).to(DEVICE)

    
    # "Setting profiler"
    if PROFILER:
        on_trace_ready = torch.profiler.tensorboard_trace_handler(PROFILER_LOGS)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            theta = ActorCritic(state_dim, action_dim).to(DEVICE)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            train(theta)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        #prof.export_chrome_trace(os.path.join(PROFILER_LOGS,"trace.json"))
    else:
        for iteration in range(NUM_META_ITERATION):
            start_time = time.time()
            print(f"Iteration {iteration + 1}/{NUM_META_ITERATION}")

            # Step 3: tasks 샘플링
            sample_task_batch = random.sample(task_batch, 5)
            sample_test_batch = random.sample(test_batch, 5)

             # --- Inner/Outer 루프 프로파일링 ---
            inner_optim = []
            post_updated_trajectories = []

            for n in range(META_PARAMETER_UPDATE):
                meta_gradients = []
                for task_idx, task in enumerate(sample_task_batch):
                    env.reset()
                    env.scenario = task

                    if n == 0:

                        inner = inner_loop(
                            state_dim, action_dim, theta, env, 1, ALPHA
                        )
                        theta_old, post_traj = inner._first_optimization()
                        inner_optim.append(OuterLoopOptimizer(ALPHA, theta_old))
                        post_updated_trajectories.append(post_traj)


                        grad = inner_optim[task_idx].optimizer(
                            post_updated_trajectories[task_idx],
                            theta, state_dim, action_dim
                        )
                        meta_gradients.append(grad)

                # meta-update

                    total_gradients = [
                        beta * sum(layer_grads)
                        for layer_grads in zip(*meta_gradients)
                    ]
                    for param, grad in zip(theta.parameters(), total_gradients):
                        param.data.add_(grad)

                # 테스트 평가
                total_reward = 0.0
                for task in sample_test_batch:
                    env.reset()
                    env.scenario = task
                    test_model = inner_loop(state_dim, action_dim, theta, env, 1, ALPHA)
                    _, reward = test_model.simulation(theta)
                    total_reward += reward

                avg_reward = total_reward / len(sample_test_batch)
                print("Average_test_Reward:", avg_reward)
                writer.add_scalar("Average_test_Reward", avg_reward, iteration)

                print(f"Iteration time: {time.time() - start_time:.2f}s")

                # 프로파일러 스텝

        print("Training & Profiling completed!")

        if SAVE_MODEL:
            torch.save(theta, os.path.join(SAVED_MODEL_PATH, 'meta_policy_promp.pth'))