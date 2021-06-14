from pettingzoo.sisl import multiwalker_v0
import wandb

env = multiwalker_v0.env()
obs = env.reset()

while True:
    for agent in env.agent_iter():
        reward, done, info = env.last()
        action = env.action_spaces[agent].sample()
        obs = env.step(action)
        env.render()
        if done:
            obs = env.reset()
env.close()
