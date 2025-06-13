import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import time
from torch.optim.lr_scheduler import LambdaLR
from config_RL import *
from config_SimPy import *
from ProMP_inner import *
from model import *

class OuterLoopOptimizer:
    def __init__(self, alpha, inner_policy):
        """
        Initialize the outer loop optimizer for ProMP.

        Args:
            alpha (float): Meta-optimizer learning rate.
            inner_policy (nn.Module): Policy network after inner adaptation (π_{θ_α}).
        """
        self.alpha = alpha
        self.clip_epsilon = CLIP_EPSILON  # PPO 클리핑 임계값
        self.gamma = GAMMA  # 할인 계수
        self.vf_coef = VF_COEF  # 가치 함수 손실 가중치
        self.ent_coef = VF_COEF  # 엔트로피 보상 계수
        self.inner_policy = inner_policy  # 내부 루프 정책 참조
        self.device = torch.device(DEVICE)
        self.gae_lambda = GAE_LAMBDA
    
    
    def _compute_gae(self, rewards, values, gamma, lambda_):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards (Tensor): Rewards from trajectory [batch_size].
            values (Tensor): State value estimates [batch_size].
            gamma (float): Discount factor.
            lambda_ (float): GAE smoothing parameter.

        Returns:
            Tensor: Computed advantage estimates [batch_size].
        """
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i] - values[i - 1] if i > 0 else 0
            gae = delta + gamma * lambda_ * gae
            advantages[i] = gae
        return advantages
    
    def optimizer(self, transitions, meta_policy, state_dim, action_dim):
        """
        Perform meta-policy update using post-adaptation transitions.

        Args:
            transitions (tuple): (states, actions, rewards, next_states, dones, log_probs_inner)
            meta_policy (nn.Module): Meta-policy to be updated.
            state_dim (int): Dimension of state space.
            action_dim (list): List of action dimensions per agent/component.
        """
        meta_policy_copy = ActorCritic(state_dim=state_dim, action_dims= action_dim).to(DEVICE)
        meta_policy_copy.load_state_dict(meta_policy.state_dict())
        optimizer = optim.Adam(meta_policy_copy.parameters(), lr=self.alpha)
        # 전이 데이터 텐서 변환
        states, actions, rewards, next_states, dones, log_probs_inner = zip(*transitions)
        
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        log_probs_inner = torch.tensor(np.array(log_probs_inner), dtype=torch.float32, device=self.device)

        # 가치 함수 계산
        _, values = self.inner_policy(states)
        _, next_values = self.inner_policy(next_states)
        not_dones = (1 - dones).unsqueeze(1)  # 종료 상태 마스킹
        next_values = (next_values * not_dones).clone()  # 그래디언트 흐름 차단

        # GAE 기반 어드밴티지 계산
        advantages = self._compute_gae(rewards, values.detach().squeeze(), self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 정규화
        value_old = rewards + self.gamma * next_values.view(-1).detach()  # 가치 타겟

        # 새 정책의 행동 분포 계산
        action_probs_new, values_new = meta_policy_copy(states)
        log_probs_new = []
        for j, dist in enumerate(action_probs_new):
            categorical_dist = Categorical(dist)
            log_probs_new.append(categorical_dist.log_prob(actions[:,j]))
        log_probs_new = torch.sum(torch.stack(log_probs_new), dim=0)
        
        # PPO 클리핑 손실 계산
        ratio = torch.exp(log_probs_new - log_probs_inner)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 총 손실 계산
        value_loss = nn.MSELoss()(values_new.view(-1), value_old)
        entropy = torch.stack([Categorical(dist).entropy().mean() for dist in action_probs_new]).mean()
        loss = policy_loss + self.ent_coef * (-entropy) + self.vf_coef * value_loss

        # 역전파 및 파라미터 업데이트
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(meta_policy_copy.parameters(), max_norm=0.5)
        optimizer.step()

        return self._cal_J_ProMP_gradient(transitions, theta_prime = meta_policy_copy, log_probs_inner = log_probs_inner)


    def _cal_J_ProMP_gradient(self, transitions, theta_prime, log_probs_inner):
        """
        Calculate ProMP objective gradient for meta-update.

        Args:
            transitions (tuple): Post-adaptation transition data.
            theta_prime (nn.Module): Adapted policy parameters (θ').
            log_probs_inner (Tensor): Log-probabilities from inner policy π_{θ_α}.

        Returns:
            tuple: Gradient of ProMP objective.
        """
        states, actions, rewards, next_states, dones, log_probs_inner = zip(*transitions)
        
        # ∇θJ^ProMP(θ) 계산
        gradient = torch.autograd.grad(
            outputs = self._cal_J_ProMP(transitions, theta_prime, log_probs_inner),
            inputs = theta_prime.parameters(),
            retain_graph=False,
            create_graph=False
        )

        return gradient
    def _cal_J_ProMP(self, transitions, theta_prime, log_probs_inner):
        """
        Calculate ProMP objective (clipped policy loss and KL penalty).

        Args:
            transitions (tuple): Post-adaptation transition data.
            theta_prime (nn.Module): Adapted policy parameters (θ').
            log_probs_inner (Tensor): Log-probabilities from inner policy π_{θ_α}.

        Returns:
            Tensor: ProMP objective value.
        """
        states, actions, rewards, next_states, dones, _ = zip(*transitions)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        log_probs_inner = torch.tensor(np.array(log_probs_inner), dtype=torch.float32, device=self.device)
        # Compute advantages using updated policy θ'
        _, values = theta_prime(states)
        _, next_values = theta_prime(next_states)
        advantages = rewards + (1 - dones) * self.gamma * next_values - values

        # Get action probabilities from θ'
        action_probs, _ = theta_prime(states)
        old_action_probs, _ = self.inner_policy(states)
        log_probs_theta_prime = []
        for j, dist in enumerate(action_probs):
            categorical_dist = Categorical(dist)
            log_probs_theta_prime.append(categorical_dist.log_prob(actions[:, j]))
        log_probs_theta_prime = torch.sum(torch.stack(log_probs_theta_prime), dim=0)


        # Compute importance ratio for clipped objective
        ratio = torch.exp(log_probs_theta_prime - log_probs_inner)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        # J_T^CLIP(θ') - 클리핑 정책 목적함수
        clipped_policy_loss = torch.min(surr1, surr2).mean()

        ##############################################
        # KL 발산 계산 추가 부분 (논문 식 13 구현)
        ##############################################
        # 이전 정책(πθo)과 새 정책(πθ)의 확률 분포 추출
        old_action_probs, _ = self.inner_policy(states)
        action_probs, _ = theta_prime(states)
        
        # 각 행동 차원별 KL 계산 후 평균
        kl_div = torch.distributions.kl.kl_divergence(
            Categorical(torch.stack(old_action_probs)), 
            Categorical(torch.stack(action_probs))
        ).mean()
        
        # 최종 ProMP 목적함수 (J^ProMP = J^CLIP - η*D_KL)
        J_promp = clipped_policy_loss - 0.1 * kl_div  # η=0.1 가정
        return J_promp