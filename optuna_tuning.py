import optuna
from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
import envs.simpy_envs.promp_env as simpy_env
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.meta_trainer import Trainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder
from envs.simpy_envs.config_SimPy import *
from envs.simpy_envs.config_folders import *
import numpy as np
import tensorflow as tf
import os
from optuna import Trial, visualization
# Ensure TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()
meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

# ====== Define training main loop as a function for Optuna ======
def run_promp(config):
    tf.compat.v1.reset_default_graph()
    set_seed(config['seed'])
    baseline = globals()[config['baseline']]()  # instantiate baseline
    env = simpy_env.MetaEnv()
    policy = MetaGaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=config['meta_batch_size'],
        hidden_sizes=config['hidden_sizes'],
    )
    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )
    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
    )
    algo = ProMP(
        policy=policy,
        inner_lr=config['inner_lr'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        learning_rate=config['learning_rate'],
        num_ppo_steps=config['num_promp_steps'],
        clip_eps=config['clip_eps'],
        target_inner_step=config['target_inner_step'],
        init_inner_kl_penalty=config['init_inner_kl_penalty'],
        adaptive_inner_kl_penalty=config['adaptive_inner_kl_penalty'],
    )
    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        tensor_log=TENSORFLOW_LOGS,
        save_folder=SAVED_MODEL_PATH,
        num_inner_grad_steps=config['num_inner_grad_steps']
    )
    trainer.train()
    # --- Training, get average reward as metric ---
    last_avg_reward = trainer.reward
    print("="*30)
    print(last_avg_reward)
    print("="*30)
    return last_avg_reward

# ====== Optuna Objective Function ======
def objective(trial):
    # Search space for two hyperparameters
    inner_lr = trial.suggest_loguniform('inner_lr', 1e-3, 0.1)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    config = {
        'seed': 1,
        'baseline': 'LinearFeatureBaseline',
        'env': 'HalfCheetahRandDirecEnv',  # Not using this parameter
        'rollouts_per_meta_task': 20,
        'max_path_length': SIM_TIME,
        'parallel': True,
        'discount': 0.99,
        'gae_lambda': 1,
        'normalize_adv': True,
        'hidden_sizes': (64, 64, 64),
        'learn_std': True,
        'inner_lr': inner_lr,
        'learning_rate': learning_rate,
        'num_promp_steps': 5,
        'clip_eps': 0.3,
        'target_inner_step': 0.01,
        'init_inner_kl_penalty': 5e-4,
        'adaptive_inner_kl_penalty': False,
        'n_itr': 301,
        'meta_batch_size': 5,
        'num_inner_grad_steps': 1,
    }
    # Run your training loop and return best metric (higher is better by default)
    avg_reward = run_promp(config)
    return avg_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print("="*30)
    print("Best params:", study.best_params)
    print("Best value:", study.best_value)
    print("="*30)

    # === Optuna 결과 시각화 ===
    # Optimization History Plot
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_html(os.path.join(HYPERPARAMETER_LOG,"optuna_optimization_history.html"))  # 파일로 저장
    # fig1.show()  # 주피터 환경이면 이거 사용

    # Parallel Coordinate Plot
    fig2 = optuna.visualization.plot_parallel_coordinate(study)
    fig2.write_html(os.path.join(HYPERPARAMETER_LOG,"optuna_parallel_coordinate.html"))

    # Contour Plot
    fig3 = optuna.visualization.plot_contour(study)
    fig3.write_html(os.path.join(HYPERPARAMETER_LOG,"optuna_contour.html"))

    print("그래프가 HTML 파일로 저장되었습니다. 브라우저로 확인하세요.")
