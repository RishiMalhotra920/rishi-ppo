import gymnasium as gym

# Create the CliffWalking environment
def make_env(render_mode):
    env = gym.make("CliffWalking-v0", render_mode=render_mode)
    return env
