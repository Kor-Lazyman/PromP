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

class OuterLoopOptimizer:
    def __init__(self, alpha, inner_policy):
        """ProMP outer-loop optimizer initialization
        
        Args:
            alpha (float): Meta-optimizer learning rate (α in equation 13)
            inner_policy (nn.Module): Inner-loop policy network (π_θ_α)
        """
        self.alpha = alpha
        self.clip_epsilon = CLIP_EPSILON  # PPO clipping threshold
        self.gamma = GAMMA  # Discount factor
        self.vf_coef = VF_COEF  # Value function loss coefficient
        self.ent_coef = VF_COEF  # Entropy bonus coefficient
        self.kl_coef = 0.1  # KL penalty coefficient (η in equation 13)
        self.inner_policy = inner_policy  # Reference to inner-loop policy
    
    def _compute_gae(self, rewards, values, gamma, lambda_):
        """Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards (Tensor): Environment rewards [batch_size]
            values (Tensor): State value estimates [batch_size]
            gamma (float): Temporal discount factor
            lambda_ (float): GAE smoothing parameter
            
        Returns:
            Tensor: Computed advantages [batch_size]
        """
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i] - values[i - 1] if i > 0 else 0
            gae = delta + gamma * lambda_ * gae
            advantages[i] = gae
        return advantages
    
    def compute_J_LR(self, states, actions, advantages, theta, theta_alpha_log_probs):
        """Compute Likelihood Ratio objective J_T^LR (Equation 12)
        
        Args:
            states (Tensor): Environment states [batch_size, state_dim]
            actions (Tensor): Selected actions [batch_size]
            advantages (Tensor): Advantage estimates A^π_α [batch_size]
            theta (nn.Module): Current meta-policy parameters
            theta_alpha_log_probs (Tensor): Log probabilities from inner policy π_θ_α
            
        Returns:
            Tensor: Likelihood ratio objective value
        """
        # Get action probabilities from current meta-policy π_θ
        action_probs = theta(states)
        dist = Categorical(action_probs)
        log_probs_theta = dist.log_prob(actions)
        
        # Compute importance ratio: π_θ(a_t|s_t) / π_θ_α(a_t|s_t)
        ratio = torch.exp(log_probs_theta - theta_alpha_log_probs)
        
        # J_T^LR = E[∑ ratio * advantages] (Equation 12)
        J_LR = (ratio * advantages).mean()
        
        return J_LR
    
    def optimizer(self, transitions, meta_policy):
        """Perform meta-policy update using ProMP objective
        
        Args:
            transitions (tuple): (states, actions, rewards, next_states, dones, log_probs_inner)
            meta_policy (nn.Module): Meta-policy to be updated (θ)
            
        Variable descriptions:
            states (Tensor): Environment states [batch_size, state_dim]
            actions (Tensor): Selected actions [batch_size]
            rewards (Tensor): Environment rewards [batch_size]
            log_probs_inner (Tensor): Inner policy log probabilities π_θ_α [batch_size]
        """
        # Store original meta-policy parameters (θ)
        original_theta = {name: param.clone() for name, param in meta_policy.named_parameters()}
        
        # Convert transition data to tensors
        states, actions, rewards, next_states, dones, log_probs_inner = zip(*transitions)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
        log_probs_inner = torch.tensor(np.array(log_probs_inner), dtype=torch.float32, device=self.device)

        # Compute value estimates using inner policy
        _, values = self.inner_policy(states)
        _, next_values = self.inner_policy(next_states)
        not_dones = (1 - dones).unsqueeze(1)  # Terminal state masking
        next_values = (next_values * not_dones).clone()  # Gradient flow blocking

        # Compute GAE-based advantages
        advantages = self._compute_gae(rewards, values.detach().squeeze(), self.gamma, self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalization

        # Step 1: Compute gradient of J_T^LR and update θ to θ'
        J_LR = self.compute_J_LR(states, actions, advantages, meta_policy, log_probs_inner)
        
        # Compute ∇_θ J_T^LR(θ)
        meta_policy.zero_grad()
        J_LR.backward(retain_graph=True)
        
        # Update θ' = θ + α ∇_θ J_T^LR(θ) (Equation 13)
        with torch.no_grad():
            for name, param in meta_policy.named_parameters():
                if param.grad is not None:
                    param.data += self.alpha * param.grad
        
        # Step 2: Compute ProMP objective J^ProMP(θ) (Equation 13)
        policy_loss, kl_div = self.cal_J_ProMP(transitions, meta_policy, log_probs_inner)
        
        # J^ProMP = J_T^CLIP(θ') - η * D_KL(π_θ_α, π_θ)
        J_ProMP = -policy_loss - self.kl_coef * kl_div  # Negative because we minimize loss
        
        # Final meta-policy update
        optimizer = optim.Adam(meta_policy.parameters(), lr=self.alpha)
        optimizer.zero_grad()
        (-J_ProMP).backward()  # Minimize negative of objective
        nn.utils.clip_grad_norm_(meta_policy.parameters(), max_norm=0.5)
        optimizer.step()

    def cal_J_ProMP(self, transitions, theta_prime, theta_alpha_log_probs):
        """Calculate ProMP performance metrics J_T^CLIP and KL divergence
        
        Args:
            transitions (tuple): Inner-loop transition data
            theta_prime (nn.Module): Updated meta-policy (θ')
            theta_alpha_log_probs (Tensor): Log probabilities from inner policy π_θ_α
            
        Returns:
            tuple: (clipped_policy_loss, kl_divergence)
            
        Variable descriptions:
            theta_prime: Updated policy parameters θ' after gradient step
            advantages: Advantage estimates A^π_α(s_t, a_t)
        """
        states, actions, rewards, next_states, dones, _ = transitions
        
        # Compute advantages using updated policy θ'
        values = theta_prime.value_net(states)
        next_values = theta_prime.value_net(next_states)
        advantages = rewards + (1 - dones) * self.gamma * next_values - values
        
        # Get action probabilities from θ'
        action_probs = theta_prime(states)
        old_action_probs = self.inner_policy(states)
        dist = Categorical(action_probs)
        log_probs_theta_prime = dist.log_prob(actions)
        
        # Compute importance ratio for clipped objective
        ratio = torch.exp(log_probs_theta_prime - theta_alpha_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        # J_T^CLIP(θ') - clipped policy objective
        clipped_policy_loss = torch.min(surr1, surr2).mean()
        
        # D_KL(π_θ_α, π_θ) - KL divergence penalty term
        kl_div = torch.distributions.kl.kl_divergence(
            Categorical(old_action_probs),  # π_θ_α
            Categorical(action_probs)  # π_θ
        ).mean()
        
        return clipped_policy_loss, kl_div
