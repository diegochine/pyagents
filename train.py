import os
from collections import deque

import numpy as np
import gym
import config as cfg
from agent import DQNAgent

env = gym.make('CartPole-v0')
state_size = (1, env.observation_space.shape[0])
action_size = env.action_space.n


if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)


agent = DQNAgent(state_size, action_size, epsilon=1.0)
# agent.load(cfg.OUTPUT_DIR + '/best.hdf5')
scores = deque(maxlen=100)

for episode in range(cfg.N_EPISODES):

    state = env.reset()
    state = np.reshape(state, agent.state_shape)
    step = 0
    done = False

    while not done and step < cfg.MAX_STEPS:
        # env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        reward = reward if not done else -100
        next_state = np.reshape(next_state, agent.state_shape)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            scores.append(step)
            print(f'EPISODE: {episode:4d}/{cfg.N_EPISODES:4d}, SCORE: {np.mean(scores):3.0f}, EPS: {agent.epsilon:.2f}')
        step += 1

    agent.replay(cfg.BATCH_SIZE)

    if (episode % 500) == 0:
        agent.save(cfg.OUTPUT_DIR + f"/ep{episode}.hdf5")

agent.save(cfg.OUTPUT_DIR + "endrun.hdf5")
