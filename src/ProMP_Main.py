import torch
from ProMP_outer import ProMPOuter
from ProMP_inner import ProMPInner
from utils.scenarios import *
from GymWrapper import GymInterface 
from config_SimPy import *
from config_RL import *
from config_folders import *
import time 
from torch.utils.tensorboard import SummaryWriter

def test_model(env, tasks, all_inner_thetas, device, step):
    total_reward = []
    for task, theta in zip(tasks, all_inner_thetas):
        env.scenario = task
        state = env.reset()
        temp_reward = 0

        for day in range(SIM_TIME):
            action, log_prob = select_action(state, theta, device)
            next_state, reward, done, info = env.step(action)
            state = next_state
            temp_reward += reward
        total_reward.append(temp_reward)

    print("Mean_Test_Reward: ", sum(total_reward)/len(tasks))
    writer.add_scalar("Mean_reward", sum(total_reward)/len(tasks), global_step = step)

# === 환경 기본 설정 ===
device = torch.device("cpu")
outer_iterations = NUM_META_ITERATION  # 예시 값
meta_parameter_updates = META_PARAMETER_UPDATE # 예시 inner loop 반복 수
penalty = KL_PENALTY
env = GymInterface()
state_dim = len(env.reset())
action_dims = [len(ACTION_SPACE) for _ in range(env.mat_count)]  # MultiDiscrete
writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)

# === Outer 객체 생성 ===
outer = ProMPOuter(state_dim, action_dims, env, penalty, device, beta=BETA)

# === 전체 task 목록 생성 ===
all_tasks = create_scenarios()
start_time = time.time()

# === outer loop 시작 ===
for step in range(outer_iterations):
    outer_start_time = time.time()
    
    # 현재 iteration에서 사용할 task 샘플링
    tasks = sample_tasks(all_tasks)
    all_inner_thetas = []
    all_mean_inner_kl = []
    
    # inner loop 반복
    for n in range(meta_parameter_updates):
        # 첫 번째 inner step인 경우
        if n == 0:
            test_model(env, tasks, all_inner_thetas, device, step)
            inner_post_updated_trajectories = []
            # 각 task에 대해 inner loop 준비
            for task in tasks:
                print(task, "start")
                inner = ProMPInner(
                    state_dim, action_dims,
                    outer.theta,  # 현재 meta-parameter
                    env,
                    task,
                    device
                )
                post_updated_trajectories, mean_kl_inner = inner.adapt()
                all_inner_thetas.append(inner.theta_old)
                all_mean_inner_kl.append(mean_kl_inner)
                inner_post_updated_trajectories.append(post_updated_trajectories)
        
        # task별 inner-loop 결과를 기반으로 meta-업데이트
        outer.update(step, all_inner_thetas, all_mean_inner_kl, tasks, inner_post_updated_trajectories)
        torch.save(outer.theta, os.path.join(SAVED_MODEL_PATH, "entire_model.pth"))
       
    print(f"Outer step learning time: {time.time()-outer_start_time}")
print(f"Learing_time: {time.time()-start_time}")

