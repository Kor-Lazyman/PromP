from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
import envs.simpy_envs.promp_env_nonstationary as simpy_env
from meta_policy_search.meta_algos.inner_test import VPGMAML
from meta_policy_search.few_shot_trainer_nonstationary import Trainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder
from envs.simpy_envs.config_SimPy import *
from envs.simpy_envs.config_folders import *
from envs.simpy_envs.scenarios import *
import numpy as np
import tensorflow as tf
import os
import time
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import csv
tf.compat.v1.disable_eager_execution()
meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def reset_classes(config, model, tasks):
    env = simpy_env.MetaEnv(tasks) # apply simpy_env wrapper to env
    if model == "ProMP":
        name = "ppo_maml" # ProMP에 등록된 Node 이름
    elif model == "VPG_MAML":
        name = "vpg_maml" # VPG_maml에 등록된 Node 이름
    else:
        name = "random"
    set_seed(config['seed'])
    
    baseline =  globals()[config['baseline']]() #instantiate baseline
    # Policy 클래스 생성
    policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )
    # Sampler 클래스 생성
    sampler_for_learning = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )
    sampler_for_test = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['Num_Test_Eps'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )
    # Sample전처리 클래스 생성
    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
    )
    # inner_Test용 graph 생성
    algo = VPGMAML(
        name= name,
        policy=policy,
        inner_type=config['inner_type'],
        inner_lr=config['inner_lr'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        )
    # Trainer 설정(inner adaption용)
    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler_for_learning = sampler_for_learning,
        sampler_for_test = sampler_for_test,
        sample_processor = sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps']
    )
    return trainer
def main(config):
    start = time.time()
    num_scenarios = 10
    tasks = []
    # task sampling
    if STATIONARY:
        for x in range(num_scenarios):
            sampled_scenario = random.sample(create_scenarios(), 1)[0]
            tasks.append(sampled_scenario)
    else:
        for x in range(num_scenarios*3):
            sampled_scenario = random.sample(create_scenarios(), 1)[0]
            tasks.append(sampled_scenario)
    
    # 각 task별 reward 수집
    rewards_by_task = {
        "VPG_MAML_Mean":[],
        "VPG_MAML_STD":[],
        "ProMP_Mean":[],
        "ProMP_STD": [],
        "Random_Mean": [],
        "Random_STD": []
    }
    actions_by_shots_before = {
        "VPG_MAML":{},
        #"VPG_MAML_STD":[],
        "ProMP":{},
        #"ProMP_STD": [],
        "Random": {}
        #"Random_STD": []
    }
    actions_by_shots_after = {
        "VPG_MAML":{},
        #"VPG_MAML_STD":[],
        "ProMP":{},
        #"ProMP_STD": [],
        "Random":{}
        #"Random_STD": []
    }
    # model들 지정
    model_type = ["ProMP", "VPG_MAML", "Random"]
    # model path 기본값 설정
    model_path = False
    for model in model_type:
        reward_by_shots= []
        # load 모델 위치 설정
        if model == "VPG_MAML":
            model_path = os.path.join(f"envs/Saved_Model/Train_45_maml_shortage_200", "model")
        elif model == "ProMP":
            model_path = os.path.join(f"envs/Saved_Model/Train_45_promp_shortage_200", "model")
        else:
            # random 파라미터는 위치가 없기 때문에 false로
            model_path = False
        task_num = 0
        for action in range(MAT_COUNT):
            actions_by_shots_before[model][action] = []
            actions_by_shots_after[model][action] = []
        #tasks[0]["DEMAND"] = {"Dist_Type": "UNIFORM", "min": 12, "max": 18}
        #tasks[1]["DEMAND"] = {"Dist_Type": "UNIFORM", "min": 8, "max": 11}
        #tasks[2]["DEMAND"] = {"Dist_Type": "UNIFORM", "min": 16, "max": 18}
        # 학습 진행
        for scenario_id in range(num_scenarios):
            print("="*10,f"Task {task_num+1}/10 Started","="*10)
            # 클래스 초기화
            if STATIONARY:
                trainer = reset_classes(config, model, [tasks[scenario_id]])
            else:
                trainer = reset_classes(config, model, tasks[3*scenario_id: 3*scenario_id+3])
            
            # 학습 후 데이터 추출
            reward_lst, before, after = trainer.train(model_path)
        #    print(after)
            for action in before:
                for i in range(len(action)):
                    action[i] = min(max(np.round(action[i]),0),10)
                    actions_by_shots_before[model][i].append(action[i])

            for action in after:
                for i in range(len(action)):
                    action[i] = min(max(np.round(action[i]),0),10)
                    actions_by_shots_after[model][i].append(action[i])
            # 리워드(10개의 shot에 대한 데이터)
            reward_by_shots.append(reward_lst)

            trainer.env.reset()
                   
            # tf의 graph 제거(network 초기화)
            tf.compat.v1.reset_default_graph()
            task_num += 1
        # 평균, 표준편차 계산
        for i in range(len(reward_by_shots[0])): # shot 수만큼
            temp_mean = []
            for j in range(len(reward_by_shots)):
                temp_mean.append(reward_by_shots[j][i])#i번 shot에 대한 리스트
            rewards_by_task[f"{model}_Mean"].append(sum(temp_mean)/num_scenarios) #i번 shot에 대한 평균
            
            #rewards_by_task[f"{model}_STD"].append(statistics.stdev(temp_mean))
    # 데이터 export
    df = pd.DataFrame(rewards_by_task)
    df = df.T
    df.to_csv(os.path.join(CSV_LOG,"Result.csv"))
    
    for model in model_type:
        temp_df = pd.DataFrame(actions_by_shots_after[model])
        temp_df.to_csv(os.path.join(CSV_LOG,f"{model}_after.csv"))
        temp_df = pd.DataFrame(actions_by_shots_before[model])
        temp_df.to_csv(os.path.join(CSV_LOG,f"{model}_before.csv"))

    print("LR-Time:", time.time()-start)
if __name__=="__main__":
    idx = int(time.time())
    # Trpo_Config
    config = {
        'seed': 1,

        'baseline': 'LinearFeatureBaseline',

        'env': 'HalfCheetahRandDirecEnv', # Not using this parameter

        # sampler config
        'rollouts_per_meta_task': 20, # number trajectorys for adapting inner loop,
        'Num_Test_Eps': 1,
        'max_path_length': SIM_TIME,
        'parallel': True, # Multi_processing

        # sample processor config
        'discount': 0.99,
        'gae_lambda': 1,
        'normalize_adv': True,

        # policy config
        'hidden_sizes': (64, 64, 64),
        'learn_std': True, # whether to learn the standard deviation of the gaussian policy

        # ProMP config
        'inner_lr': 0.1, # adaptation step size
        'n_itr': 11, # number of overall training iterations
        'meta_batch_size': 1, # number of sampled meta-tasks per iterations
        'num_inner_grad_steps': 1, # number of inner / adaptation gradient steps
        'inner_type' : 'log_likelihood', # type of inner loss function used
    }
    
    # start the actual algorithm
    main(config)