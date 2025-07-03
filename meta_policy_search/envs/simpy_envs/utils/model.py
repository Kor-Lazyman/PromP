from config_SimPy import *
from config_RL import *
import torch
import torch.nn as nn
import itertools
import random
import numpy as np
from torch.distributions import Categorical
from collections import defaultdict
from utils.scenarios import *

# 1. Task Sampling Related

def likelihood(theta_old, theta, trajectories, env, device, loop):
    """
    Computes the likelihood for the inner or outer loop in the ProMP algorithm.
    The likelihood is computed based on the log probabilities of actions from the
    given policy and the advantage estimates.
    
    Args:
        theta_old: The previous model's parameters (used in the inner loop).
        theta: The current model's parameters.
        trajectories: A list of trajectories (state-action-reward sequences).
        env: The environment used for simulations.
        device: The device (CPU/GPU) where computations should be performed.
        loop: Specifies whether it's the 'inner' or 'outer' loop.

    Returns:
        surr_obj: The surrogate objective function for policy optimization.
    """
    theta.train()
    
    if loop == "inner":
        # Unzip trajectories for the inner loop
        states, actions_meta, rewards_meta, next_states_meta, dones_meta, log_probs_meta = unzip_trajectories(trajectories, device)
        states_old, actions_old, rewards_old, next_states_old, dones_old, log_probs_old = unzip_trajectories(collect_trajectories(states, env, theta_old, device), device)
        # Compute action probabilities for the current model
        action_probs_meta = theta(states)
    elif loop == "outer":
        # Unzip trajectories for the outer loop
        states, actions_old, rewards_old, next_states_old, dones_old, log_probs_old = unzip_trajectories(trajectories, device)
        states_meta, actions_meta, rewards_meta, next_states_meta, dones_meta, log_probs_meta = unzip_trajectories(collect_trajectories(states, env, theta, device), device)
        # Compute action probabilities for the current model
        action_probs_meta = theta(states)
    log_probs_meta = []

    # Calculate log probabilities for each action in the current model
    for j, dist in enumerate(action_probs_meta):
        categorical_dist = Categorical(dist)
        log_probs_meta.append(categorical_dist.log_prob(actions_meta[:, j]))
    log_probs_meta = torch.sum(torch.stack(log_probs_meta), dim=0)
    
    not_dones = (1 - dones_old).unsqueeze(1)

    # Calculate values from episodes
    values_old = cal_value_from_episodes(rewards_old, device)
    values_new = cal_value_from_episodes(rewards_meta, device)

    advantages = _compute_gae(rewards_old, values_old.squeeze(), GAMMA, GAE_LAMBDA, device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    if loop == "outer":
        # Compute the surrogate objective for the outer loop
        surr1 = torch.exp(log_probs_meta - log_probs_old) * advantages
        surr2 = torch.clamp(torch.exp(log_probs_meta - log_probs_old), 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
        surr_obj = torch.min(surr1, surr2).mean()
    else:
        # Compute the surrogate objective for the inner loop
        ratio = torch.exp(log_probs_meta - log_probs_old)
        value_loss = nn.MSELoss()(values_new.view(-1), values_old)
        surr_obj = torch.mean(ratio * advantages)

    return surr_obj


def _compute_gae(rewards, values, gamma, lambda_, device):
    """
    Computes Generalized Advantage Estimation (GAE) for PPO.
    
    This method computes the advantage estimates which are used for policy updates in PPO.
    
    Args:
        rewards: Rewards obtained from environment.
        values: Estimated values of the states.
        gamma: Discount factor.
        lambda_: Smoothing factor for GAE.

    Returns:
        torch.Tensor: Computed advantage estimates.
    """
    advantages = torch.zeros_like(rewards, device=device)
    gae = 0
    for i in reversed(range(len(rewards))):
        # Compute temporal difference (TD) error (delta)
        delta = rewards[i] + gamma * values[i] - values[i - 1] if i > 0 else 0
        gae = delta + gamma * lambda_ * gae  # Accumulate GAE
        advantages[i] = gae
    return advantages


# 6. KL Divergence Calculation

def kl_divergence(old_model, meta_model, trajectories, device):
    """
    Computes the KL divergence between the policies of two models (old and meta-model).
    
    This is used to measure how much the new model's policy diverges from the old policy,
    which can be used for regularization during training.

    Args:
        old_model: The old model used for generating behavior (before update).
        meta_model: The meta model used for current policy.
        trajectories: A list of trajectories (state-action-reward sequences).
        device: The device (CPU/GPU) where computations should be performed.

    Returns:
        mean_kl: The average KL divergence across all states.
    """
    old_model.eval()
    meta_model.eval()

    states, _, _, _, _, _ = zip(*trajectories)
    states = torch.tensor(np.array(states), dtype=torch.float32, device=device)

    kl_values = []

    with torch.no_grad():
        for state in states:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            old_logits = old_model(state)
            meta_logits = meta_model(state)

            # Compute the Categorical distributions for both models
            old_dist = Categorical(torch.stack(old_logits))
            meta_dist = Categorical(torch.stack(meta_logits))

            # Compute the KL divergence between the two distributions
            kl = torch.distributions.kl_divergence(old_dist, meta_dist)
            kl_values.append(kl)

    # Compute the mean KL divergence across all states
    mean_kl = torch.stack(kl_values).mean()

    return mean_kl

def cal_value_from_episodes(rewards, device):
    """
    Calculates the value of each state using Monte Carlo method.
    
    This method computes the return (G_t) for each time step in an episode,
    which is the sum of future discounted rewards.

    Args:
        rewards: List of rewards obtained from the environment.
        device: The device (CPU/GPU) where computations should be performed.

    Returns:
        torch.Tensor: The calculated values for each state in the episode.
    """
    returns = torch.zeros_like(rewards, dtype=torch.float32, device=device)
    G = 0
    # Calculate the return in reverse order (backwards from the last reward)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + GAMMA * G  # Accumulate discounted reward
        returns[t] = G  # Assign return to the current time step
    return returns
