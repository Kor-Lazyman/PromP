
import torch
import torch.nn as nn
import torch.optim as optim
class ActorCritic(nn.Module):
    """
    Actor-Critic model for PPO with MultiDiscrete action space.

    Args:
        state_dim: Dimension of the state space.
        action_dims: List containing the number of discrete actions per action dimension.
        hidden_size: Number of neurons in hidden layers.
    """
    def __init__(self, state_dim, action_dims, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.action_dims = action_dims

        # Policy Network (Actor)
        self.actor_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        self.action_heads = nn.ModuleList([nn.Linear(hidden_size, dim) for dim in action_dims])  # MultiDiscrete

        # Value Network (Critic)
        self.critic_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the Actor-Critic network.

        Args:
            state: Current state of the environment.

        Returns:
            action_probs: Probability distributions for MultiDiscrete action dimensions.
            value: Estimated state value.
        """
        actor_features = self.actor_fc(state)
        action_probs = [torch.softmax(head(actor_features), dim=-1) for head in self.action_heads]  # MultiDiscrete
        value = self.critic_fc(state)
        return action_probs, value