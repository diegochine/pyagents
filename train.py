import os
from collections import deque
import numpy as np
import gym
from agents import DQNAgent
from networks import QNetwork
from memory import PrioritizedBuffer
from keras.optimizers import RMSprop
import gin


OUTPUT_DIR = './output'
BATCH_SIZE = 64
N_EPISODES = 501
MAX_STEPS = 1000
MIN_MEMORIES = 2000

env = gym.make('Asteroids-ram-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

gin.parse_config_file('config/cartpole_v0.gin')
buffer = PrioritizedBuffer()
q_net = QNetwork(state_size, action_size)
player = DQNAgent(state_size, action_size, q_network=q_net, buffer=buffer, optimizer=RMSprop(momentum=0.1), name='cartpole')
# player = DQNAgent.load('output/cartpole', epsilon=0.01, training=False)
scores = deque(maxlen=100)
player.memory_init(env, MAX_STEPS, MIN_MEMORIES)

for episode in range(N_EPISODES):

    state = env.reset()
    state = np.reshape(state, player.state_shape)
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        env.render()
        action = player.act(state)
        next_state, reward, done, info = env.step(action)
        reward = reward if not done else -100
        next_state = np.reshape(next_state, player.state_shape)
        player.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            scores.append(step)
            print(f'EPISODE: {episode:4d}/{N_EPISODES:4d}, SCORE: {np.mean(scores):3.0f}, EPS: {player.epsilon:.2f}')
        step += 1

    player.train(BATCH_SIZE)

    if (episode % 100) == 0:
        player.save(ver=episode//100)
