from ppo import ppo
from env import make_env

def train():
    env = make_env(render_mode=None)
    actor_layers = [env.observation_space.n, 10, 10, env.action_space.n]
    critic_layers = [env.observation_space.n, 10, 10, 1]
    num_epochs = 100
    num_steps_per_epoch = 1000
    actor_lr = 1e-3
    critic_lr = 1e-3
    gamma = 0.9
    lamb = 0.9
    ppo_clip = 0.2
    entropy_weight = 0.1

    ppo(env, actor_layers, critic_layers, num_epochs, num_steps_per_epoch, actor_lr, critic_lr, gamma, lamb, entropy_weight, ppo_clip)

if __name__ == "__main__":
    train()