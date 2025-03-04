import torch
import torch.nn as nn
from torch import distributions 
from datetime import datetime
import os

def mlp(layers, activation, final_activation=nn.Identity):
    pt_layers = [] 
    for i in range(len(layers)-1):
        pt_layers.append(nn.Linear(layers[i], layers[i+1]))
        pt_layers.append(activation())
    
    pt_layers.pop()
    pt_layers.append(final_activation())
    return nn.Sequential(*pt_layers)


class DiscreteActor(nn.Module):
    def __init__(self, layers, activation):
        super().__init__()
        self.policy = mlp(layers, activation)
    
    def forward(self, obs):
        logits = self.policy(obs)
        return logits
    
    def get_distribution(self, logits):
        return distributions.Categorical(logits=logits)
    

class Critic(nn.Module):
    def __init__(self, layers, activation):
        super().__init__()
        self.v = mlp(layers, activation)

    def forward(self, x):
        return self.v(x)
        

class ActorCritic:
    def __init__(self, actor_layers, critic_layers, activation):
        self.actor = DiscreteActor(actor_layers, activation)
        self.v = Critic(critic_layers, activation)

    def step(self, obs):
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32)
            act_logits = self.actor(obs)
            act_disb = self.actor.get_distribution(act_logits)
            act = act_disb.sample()
            logp = act_disb.log_prob(act)
            v = self.v(obs)
            return act.numpy(), v.numpy(), logp.numpy()
    
    def save(self, actor_optimizer, critic_optimizer, checkpoint_dir: str, epoch: str):
        # Create the directory if it doesn't exist

        torch.save({
            "policy_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.v.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict()
        }, 
        f"{checkpoint_dir}/checkpoint_{epoch}.pth"
        )

def load(path, obs_space, action_space):
    weights = torch.load(path)
    actor_layers = [obs_space, 10, 10, action_space]
    critic_layers = [obs_space, 10, 10, 1]  # Changed input from action_space to obs_space
    ac = ActorCritic(actor_layers, critic_layers, nn.ReLU)
    ac.actor.load_state_dict(weights["policy_state_dict"])
    ac.v.load_state_dict(weights["critic_state_dict"])
    return ac
        
