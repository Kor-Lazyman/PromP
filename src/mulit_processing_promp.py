from ProMP_inner import *
from ProMP_optimizer import OuterLoopOptimizer
from config_RL import *
from config_SimPy import *
from config_folders import *
from GymWrapper import *
from Def_Scenarios import *
from model import *
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
import random

if __name__ == '__main__':
    writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
    env = GymInterface()
    beta = BETA
    action_dim = [len(ACTION_SPACE) for _ in range(env.mat_count)]
    state_dim = len(env.reset())
    # Meta-parameters (already set up as mentioned)
    num_iterations = 1000
    batch_size = 16
    convergence_threshold = 1e-4
    # Initialize meta-policy parameters θ
    theta = ActorCritic(state_dim, action_dim)
    for iteration in range(num_iterations):
        start_time = time.time()
        with record_function("meta_iteration"):
            print(f"Iteration {iteration + 1}/{num_iterations}")

            # Step 3: Sample batch of tasks Ti ~ ρ(T)
            scenarios = create_scenarios()
            task_batch, test_batch = split_scenarios(scenarios)
            # Initialize gradient accumulator for meta-update
            inner_optim = []
            post_updated_trajectories = []

            sample_task_batch = random.sample(task_batch, TASK_BATCH_SIZE)
            for n in range(100):
                meta_gradients = []
                for task_idx in range(len(sample_task_batch)):
                    env.reset()
                    env.scenario = sample_task_batch[task_idx]

                    if n == 0:
                        # Step 8: Sample pre-update trajectories using current policy πθ            
                        # Step 9: Compute adapted parameters θ'0,i using inner loop
                        inner = inner_loop(state_dim, action_dim, theta.to(DEVICE), env, 1, ALPHA)
                        # Step 10: Sample post-update trajectories using adapted policy πθ'
                        theta_old, post_updated_trajectorie = inner._first_optimization()

                        inner_optim.append(OuterLoopOptimizer(ALPHA, theta_old))
                        post_updated_trajectories.append(post_updated_trajectorie)
                    
                    with record_function("outer_loop_optimizer"):
                        meta_gradients.append(
                            inner_optim[task_idx].optimizer(
                                post_updated_trajectories[task_idx], theta, state_dim, action_dim
                            )
                        )
                # Step 11: Meta-update θ ← θ + β∑Ti ∇θJ^ProMP_Ti(θ)
                # Average gradients across all tasks in the batch
                total_gradients = [
                    beta * sum(layer_grads)
                    for layer_grads in zip(*meta_gradients)
                ]
                with torch.no_grad():
                    for param, grad in zip(theta.parameters(), total_gradients):
                        param.data.add_(grad)

            # 테스트 보상 측정
            total_reward = 0
            for task_idx in range(len(test_batch)):
                env.reset()
                env.scenario = test_batch[task_idx]  # ❗이전 코드에서 task_batch 잘못 사용됨

                test_model = inner_loop(state_dim, action_dim, theta.to(DEVICE), env, 1, ALPHA)
                _, reward = test_model.simulation(theta)
                total_reward += reward
            print("Average_test_Reward:", total_reward/len(test_batch))
            writer.add_scalar("Average_test_Reward", total_reward / len(test_batch), iteration)
        print(f"Learning_Time: {time.time()-start_time}")
        # ✅ 프로파일러 step() 호출로 schedule 흐름 제어

    print("Training completed!")
    torch.save(theta, os.path.join(SAVED_MODEL_PATH, 'meta_policy_promp.pth'))