import torch
from ProMP_Model import Model
from utils.model import *
from config_RL import *
import time
class ProMPInner:
    def __init__(self, state_dim, action_dim, theta, env, task, device):
        self.device = device
        self.env = env
        self.task = task
        self.batch_size = 20
        self.epochs = 5
        self.alpha = 0.0001  # 고정 inner-loop step size

        # θ는 기준점으로 고정
        self.theta = theta
        # θ_old는 적응 대상 (업데이트될 파라미터)
        self.theta_old = Model(state_dim, action_dim).to(device)
        self.theta_old.load_state_dict(theta.state_dict())

    def adapt(self):
        param = self.theta_old.parameters
        self.env.scenario = self.task
        # 1. meta_parameter 정책 (θ)로 trajectory 수집
        pre_updated_trajectories = sample_trajectories(self.theta, self.env, self.device)
        # 2. Trajectory → 배치화
        #batches = utils.make_trajectorie_batches(pre_updated_trajectories, self.batch_size)
        # 3. Inner-loop 학습
        kl_divergences = []
        for _ in range(1):
            '''
            for batch in batches:
                surr_obj = utils.likelihood(self.theta_old, self.theta, batch, self.env, self.device, "inner")
                self.theta.zero_grad()
                surr_obj.backward()
            '''
            
            surr_obj = likelihood(self.theta_old, self.theta, pre_updated_trajectories, self.env, self.device, "inner")
            self.theta.zero_grad()
            surr_obj.backward()
            
            # 직접 gradient로 업데이트
            with torch.no_grad():
                for param, param_ref in zip(self.theta_old.parameters(), self.theta.parameters()):
                    if param_ref.grad is not None:
                        param = param_ref + self.alpha * param_ref.grad

            post_updated_trajectories = sample_trajectories(self.theta_old, self.env, self.device)
            kl_divergences.append(kl_divergence(self.theta_old, self.theta, post_updated_trajectories, self.device))
            

            
            states, actions, rewards, next_states, dones, log_probs = unzip_trajectories(post_updated_trajectories, self.device)
        print("Total_reward:", sum(rewards))
        return post_updated_trajectories, sum(kl_divergences)/len(kl_divergences)
