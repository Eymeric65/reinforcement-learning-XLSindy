import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import torch


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs,model_path=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1,np.prod(envs.action_space.shape)))

        if model_path is not None:
            self.load_model(model_path)

    def _flatten_space(self,x):
        return x.view(x.size(0), -1)
    
    def get_value(self, x):
        x=self._flatten_space(x)

        return self.critic(x)

    def get_action_and_value(self, x, action=None):

        if action is not None:
            action =self._flatten_space(action)
        
        x = self._flatten_space(x)

        action_mean = self.actor_mean(x)
        #print(action_mean.size())
        #print(self.actor_logstd.size())
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
    def load_model(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)