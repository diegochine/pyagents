import os
import numpy as np
import gym
from argparse import ArgumentParser
from pyagents.agents import DQNAgent, A2C
from pyagents.networks import QNetwork, ActorNetwork, ValueNetwork
from pyagents.memory import PrioritizedBuffer
from tensorflow.keras.optimizers import Adam
import gin
import matplotlib.pyplot as plt


@gin.configurable
def train_dqn_agent(n_episodes=1000, batch_size=64, learning_rate=0.001, steps_to_train=5,
                    max_steps=500, min_memories=5000, output_dir="./output/"):

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    buffer = PrioritizedBuffer()
    q_net = QNetwork(state_size, action_size)
    optim = Adam(learning_rate=learning_rate)
    player = DQNAgent(state_size, action_size, q_network=q_net, buffer=buffer, optimizer=optim, name='cartpole')
    # player = DQNAgent.load('output/cartpole', ver=5, epsilon=0.01, training=False)
    scores = []
    movavg100 = []
    eps_history = []
    player.memory_init(env, max_steps, min_memories)

    for episode in range(1, n_episodes+1):

        s_t = env.reset()
        s_t = np.reshape(s_t, player.state_shape)
        step = 0
        done = False

        while not done and step < max_steps:
            # env.render()
            a_t = player.act(s_t)
            s_tp1, r_t, done, info = env.step(a_t)
            r_t = r_t if not done else -100
            s_tp1 = np.reshape(s_tp1, player.state_shape)
            player.remember(state=s_t, action=a_t, reward=r_t, next_state=s_tp1, done=done)
            s_t = s_tp1

            if step % steps_to_train == 0:
                player.train(batch_size)
            step += 1

        scores.append(step)
        this_episode_score = np.mean(scores[-10:])
        movavg100.append(this_episode_score)
        eps_history.append(player.epsilon)
        if episode % 10 == 0:
            print(f'EPISODE: {episode:4d}/{n_episodes:4d}, '
                  f'SCORE: {this_episode_score:3.0f}, '
                  f'EPS: {player.epsilon:.2f}')

        player.train(batch_size)

        if (episode % 100) == 0:
            player.save(ver=episode//100)

    return scores, movavg100, eps_history


@gin.configurable
def train_a2c_agent(n_episodes=1000, learning_rate=5e-4, max_steps=250, output_dir="./output/"):

    env = gym.make('Pendulum-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    a_net = ActorNetwork(state_size, action_size, distribution='gaussian', fc_layer_params=(16, 16))
    v_net = ValueNetwork(state_size, action_size, fc_layer_params=(16, 16))
    a_opt = Adam(learning_rate=learning_rate)
    v_opt = Adam(learning_rate=learning_rate)
    player = A2C(state_size, action_size, actor=a_net, critic=v_net, actor_opt=a_opt, critic_opt=v_opt, gamma=0.5, name='pendulum')
    # player = DQNAgent.load('output/cartpole', ver=5, epsilon=0.01, training=False)
    scores = []
    movavg100 = []
    eps_history = []

    for episode in range(1, n_episodes+1):

        s_t = env.reset()
        s_t = np.reshape(s_t, player.state_shape)
        step = 0
        score = 0
        done = False

        while not done and step < max_steps:
            # env.render()
            a_t = player.act(s_t)
            s_tp1, r_t, done, info = env.step(a_t)
            s_tp1 = np.reshape(s_tp1, player.state_shape)
            player.remember(state=s_t, action=a_t, reward=r_t)
            s_t = s_tp1
            step += 1
            score += r_t

        losses = player.train()

        scores.append(score)
        this_episode_score = np.mean(scores[-2:])
        movavg100.append(this_episode_score)
        # eps_history.append(player.epsilon)
        if episode % 1 == 0:
            print(f'EPISODE: {episode:4d}/{n_episodes:4d}, SCORE: {this_episode_score:3.0f}, '
                  f'POLICY LOSS: {losses["policy_loss"]:4.2f}, CRITIC LOSS: {losses["critic_loss"]:5.2f}')

        # if (episode % 100) == 0:
        #     player.save(ver=episode//100)

    return scores, movavg100, eps_history


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training a sample agent on Gym")
    parser.add_argument('-a', '--agent', type=str, help='which agent to train, either DQN or A2C')
    parser.add_argument('-c', '--config', type=str, help='path to gin config file', default='')
    args = parser.parse_args()
    results = []
    if args.config:
        gin.parse_config_file(args.config)
    if args.agent == 'dqn':
        info = train_dqn_agent()
    elif args.agent == 'a2c':
        info = train_a2c_agent()
    else:
        raise ValueError(f'wrong agent {args.agent}')
