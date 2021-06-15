import os
import numpy as np
import gym
from argparse import ArgumentParser
from agents import DQNAgent
from networks import QNetwork
from memory import PrioritizedBuffer
from keras.optimizers import Adam
import gin
import matplotlib.pyplot as plt


@gin.configurable
def train_agent(n_episodes=1000, batch_size=64, learning_rate=0.001, steps_to_train=100,
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

        state = env.reset()
        state = np.reshape(state, player.state_shape)
        step = 0
        done = False

        while not done and step < max_steps:
            # env.render()
            action = player.act(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -100
            next_state = np.reshape(next_state, player.state_shape)
            player.remember(state, action, reward, next_state, done)
            state = next_state

            if step % steps_to_train == 0:
                player.train(batch_size)
            step += 1

        scores.append(step)
        this_episode_score = np.mean(scores[-100:])
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training agents")
    parser.add_argument('-c', '--config', nargs='+', help='path to gin config file(s) ', required=True)
    args = parser.parse_args()
    results = []
    for cfg_file in args.config:
        gin.parse_config_file(cfg_file)
        info = train_agent(n_episodes=500)
        results.append((cfg_file, info))
    ncols = 2
    nrows = len(args.config)//2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='all', constrained_layout=True)
    axes = axes.reshape(-1)
    for i, (name, result) in enumerate(results):
        ax = axes[i]
        scores, movavg_scores, eps = result
        ax.set_title(name)
        ax.plot(scores, color='b', linewidth=1, label='Per-episode scores')
        ax.plot(movavg_scores, color='k', linewidth=3, label='Moving average scores')
        ax.set_ylabel('Score', color='b')
        ax2 = axes[i].twinx()
        ax2.plot(eps, color='r', linewidth=2, label='Epsilon history')
        ax2.set_ylabel('Epsilon value', color='r')
        ax.grid()

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines = lines_1 + lines_2
        labels = labels_1 + labels_2
        ax.legend(lines, labels)

    axes[-2].set_xlabel('Episode')
    axes[-1].set_xlabel('Episode')
    plt.show()
