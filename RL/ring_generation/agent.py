import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from env import N_ATOM_FEATURES


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_dim=128, bias=True):
        super(PolicyNetwork, self).__init__()
        self.layer = nn.Linear(N_ATOM_FEATURES, hidden_dim, bias)
        self.pred_start = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias),
        )
        self.pred_end = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias),
        )

    def forward(self, input, adj):
        out = adj @ self.layer(input)
        start = self.pred_start(out).squeeze(-1)
        action_start, prob_start = self.sample(start)    # B,1 or 1
        end = self.pred_end(out).squeeze(-1)
        end = self.exclude(end, action_start)
        action_end, prob_end = self.sample(end)    # B,1 or 1
        actions = [action_start, action_end]
        probs = [prob_start, prob_end]
        return actions, probs

    def exclude(self, end, action_start):
        if len(action_start.shape) > 1:
            for i in range(end.shape[0]):
                end[i,action_start[i]] = -1e10
        else:
            end[action_start] = -1e10
        return end

    def sample(self, tensor):
        prob = torch.softmax(tensor, dim=-1)
        dist = Categorical(prob)
        action = dist.sample()
        return action, prob


class ValueNetwork(nn.Module):
    def __init__(self, hidden_dim=128, bias=True):
        super(ValueNetwork, self).__init__()
        self.layer = nn.Linear(N_ATOM_FEATURES, hidden_dim, bias)
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias),
        )

    def forward(self, input, adj):
        out = adj @ self.layer(input)
        value = self.pred(out).squeeze(-1)
        return value.sum(-1)


class ActorCritic(object):
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
