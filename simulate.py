from env import make_env
from ppo import get_one_hot
from actor_critic import load


def simulate(path):
    env = make_env("human") # or None
    ac = load(path, env.observation_space.n, env.action_space.n)

    for trial in range(10):
        terminated = False
        truncated = False
        obs, info = env.reset()

        while not (terminated or truncated):  # Check both conditions
            act, val, logp = ac.step(get_one_hot(env.observation_space.n, obs))
            obs, rew, terminated, truncated, info = env.step(act.item())
            print('rew', rew, terminated or truncated)
            
            if terminated or truncated:
                print("Done")
                
        print(f"Reward received in {trial}: {rew}")
    

if __name__ == "__main__":
    path = "checkpoints/2025-03-03_23-57-13/checkpoint_9.pth"
    simulate(path)