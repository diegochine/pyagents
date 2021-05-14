import os
from collections import deque
import numpy as np
import gym
import config as cfg
from agents import DQNAgent
from networks import QNetwork
from keras.optimizers import Adam


env = gym.make('CartPole-v0')
state_size = (1, env.observation_space.shape[0])
action_size = env.action_space.n


if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

q_net = QNetwork(state_size, action_size, fc_layer_params=(32, 32, 32))
player = DQNAgent(state_size, action_size, q_network=q_net, optimizer=Adam(learning_rate=0.01),
                  epsilon=1.0)
scores = deque(maxlen=100)

for episode in range(cfg.N_EPISODES):

    state = env.reset()
    state = np.reshape(state, player._state_shape)
    step = 0
    done = False

    while not done and step < cfg.MAX_STEPS:
        env.render()
        action = player.act(state)
        next_state, reward, done, info = env.step(action)
        reward = reward if not done else -100
        next_state = np.reshape(next_state, player._state_shape)
        player.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            scores.append(step)
            print(f'EPISODE: {episode:4d}/{cfg.N_EPISODES:4d}, SCORE: {np.mean(scores):3.0f}, EPS: {player._epsilon:.2f}')
        step += 1

    player.train(cfg.BATCH_SIZE)

    #if (episode % 500) == 0:
    #    player.save(cfg.OUTPUT_DIR + f"/ep{episode}.hdf5")

#player.save(cfg.OUTPUT_DIR + "endrun.hdf5")
