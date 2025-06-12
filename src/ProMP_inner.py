import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import time
from torch.optim.lr_scheduler import LambdaLR
from config_RL import *
from config_SimPy import *

class inner_loop:
    def __init__(self, meta_policy, env, learning_steps, alpha):
        """
        Initialize inner loop for ProMP meta-learning.

        Args:
            meta_policy (nn.Module): Meta-policy network (πθ).
            env: Environment instance for task Ti.
            learning_steps (int): Number of inner adaptation steps.
            alpha (float): Inner loop learning rate (α).
        """
        self.old_policy = meta_policy  # Meta-policy πθ
        self.env = env  # Task environment Ti
        self.gamma = GAMMA  # Discount factor
        self.clip_epsilon = CLIP_EPSILON  # PPO clipping parameter
        self.update_steps = 1
        self.gae_lambda = GAE_LAMBDA          
        self.ent_coef = ENT_COEF              
        self.vf_coef = VF_COEF                
        self.max_grad_norm = MAX_GRAD_NORM    
        self.device = torch.device(DEVICE)
        self.learning_steps = learning_steps
        self.optimizer = optim.Adam(self.old_policy.parameters(), lr=alpha)  # Inner optimizer

        # Learning rate scheduler for inner loop
        lr_lambda = lambda step: 1 - min(step, learning_steps) / (learning_steps)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
    
    def _select_action(self, state, policy):
        """
        Select action and log-probability using the given policy.

        Args:
            state: Current environment state.
            policy (nn.Module): Policy network.

        Returns:
            tuple: (actions (np.array), log_probabilities (Tensor))
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs, _ = policy(state)
        
        actions = []
        log_probs = []
        for dist in action_probs:
            categorical_dist = Categorical(dist)
            action = categorical_dist.sample()
            actions.append(action.item())
            log_probs.append(categorical_dist.log_prob(action))
        
        return np.array(actions), torch.sum(torch.stack(log_probs)) 

    def simulation(self, policy):
        """
        Sample a trajectory from the environment using the specified policy.

        Args:
            policy (nn.Module): Policy network for trajectory sampling.

        Returns:
            list: Episode transitions (state, action, reward, next_state, done, log_prob).
        """
        state = self.env.reset()
        done = False
        episode_transitions = []
        episode_reward = 0

        # Collect trajectory τ = {(st, at, rt, st+1, dt)}
        for day in range(SIM_TIME):
            action, log_prob = self._select_action(state, policy)
            next_state, reward, done, info = self.env.step(action)
            episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
            episode_reward += reward
            state = next_state
        
        return episode_transitions

    def _compute_gae(self, rewards, values, gamma, lambda_):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards (Tensor): Rewards from trajectory.
            values (Tensor): State value estimates.
            gamma (float): Discount factor.
            lambda_ (float): GAE smoothing parameter.

        Returns:
            Tensor: Computed advantage estimates.
        """
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i] - values[i - 1] if i > 0 else 0
            gae = delta + gamma * lambda_ * gae
            advantages[i] = gae
        return advantages 
    
    def _first_optimization(self):
        """
        Perform inner loop adaptation (one-step policy update).

        Steps:
            1. Sample pre-update trajectories using meta-policy.
            2. Compute policy gradient and adapt parameters.
            3. Sample post-update trajectories using adapted policy.

        Returns:
            tuple: (adapted policy, post-update trajectory transitions)
        """
        start_time = time.time()
        
        for _ in range(1):  # TODO: Remove later
            # Step 1: Sample pre-update trajectories Di = {τi} from Ti using πθ
            meta_transition = self.simulation(self.old_policy)

            # Extract trajectory components
            states, actions, rewards, next_states, dones, log_probs_meta = zip(*meta_transition)
            
            # Convert to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
            dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)
            log_probs_meta = torch.tensor(np.array(log_probs_meta), dtype=torch.float32, device=self.device)  # Fixed variable name
            
            # Compute value estimates using current policy πθ
            _, values = self.old_policy(states)
            _, next_values = self.old_policy(next_states)
            not_dones = (1 - dones).unsqueeze(1)  # Terminal state masking
            next_values = (next_values * not_dones).clone()
            
            # Compute value targets for advantage estimation
            value_targets = rewards + self.gamma * next_values.view(-1).detach()

            # Compute advantages using GAE
            advantages = self._compute_gae(value_targets, values.detach().squeeze(), self.gamma, self.gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
            value_meta = value_targets + self.gamma * next_values.view(-1).detach()

            # Mini-batch sampling for stable updates
            batch_size = BATCH_SIZE  
            dataset_size = len(states)
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            # Step 2: Parameter adaptation via gradient updates
            for i in range(0, dataset_size, batch_size):
                # Sample mini-batch from pre-update trajectories
                batch_indices = indices[i : i + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_meta = log_probs_meta[batch_indices].detach().clone()
                batch_value_meta = value_meta[batch_indices].detach().clone()

                # Compute current policy outputs
                action_probs_old, values_old = self.old_policy(batch_states)
                
                # Calculate log probabilities for current policy
                log_probs_old = []
                for j, dist in enumerate(action_probs_old):
                    categorical_dist = Categorical(dist)
                    log_probs_old.append(categorical_dist.log_prob(batch_actions[:, j]))
                
                log_probs_old = torch.sum(torch.stack(log_probs_old), dim=0)
                
                # PPO clipped objective (approximates likelihood ratio gradient)
                ratio = torch.exp(batch_log_probs_meta - log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = nn.MSELoss()(values_old.view(-1), batch_value_meta)
                
                # Entropy bonus for exploration
                entropy = torch.stack([
                    Categorical(dist).entropy().mean() for dist in action_probs_old
                ]).mean()
                entropy_loss = -entropy
                
                # Total loss: L = L_policy + c1*L_value + c2*L_entropy
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                
                # Gradient step: θ'α,i ← θ + α ∇θ J^LR(θ)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)  
                nn.utils.clip_grad_norm_(self.old_policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                self.learn_time = time.time() - start_time

            # Update learning rate and clipping
            self.scheduler.step()
            self.clip_epsilon = max(0.1, self.clip_epsilon * 0.995)

        # Step 3: Sample post-update trajectories D'i = {τ'i} using πθ'α,i
        new_transition = self.simulation(self.old_policy)
        
        return self.old_policy, new_transition
