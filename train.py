from ppo import ppo
from env import make_env

def train():
    env = make_env(render_mode=None)
    actor_layers = [env.observation_space.n, 10, 10, env.action_space.n]
    critic_layers = [env.observation_space.n, 10, 10, 1]
    iterations = 1000
    num_epochs = 10
    num_steps_per_epoch = 50
    actor_lr = 1e-3
    critic_lr = 1e-2
    gamma = 0.9
    lamb = 0.9
    ppo_clip = 0.2
    entropy_weight = 0.5

    ppo(env, actor_layers, critic_layers, iterations, num_epochs, num_steps_per_epoch, actor_lr, critic_lr, gamma, lamb, entropy_weight, ppo_clip)

if __name__ == "__main__":
    train()