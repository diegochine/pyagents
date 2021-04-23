import os
import numpy as np
import gym
import config as cfg
from agent import DQNAgent

env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n


if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)


agent = DQNAgent(state_size, action_size)
# agent.load(cfg.OUTPUT_DIR + '/endrun.hdf5')

for episode in range(cfg.N_EPISODES):

    state = env.reset()
    state = np.reshape(state, (1, agent.state_shape))
    step = 0
    done = False

    while not done and step < 1000:
        # env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        reward = reward if not done else -100
        next_state = np.reshape(next_state, (1, agent.state_shape))
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print(f'EPISODE: {episode:4d}/{cfg.N_EPISODES:4d}, SCORE: {step:4d}, EPS: {agent.epsilon}')
        step += 1

    agent.replay(cfg.BATCH_SIZE)

    if (episode % 1000) == 0:
        agent.save(cfg.OUTPUT_DIR + f"/ep{episode}.hdf5")

agent.save(cfg.OUTPUT_DIR + "endrun.hdf5")
