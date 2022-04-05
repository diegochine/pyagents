import os
import random

import numpy as np
import gym
from argparse import ArgumentParser

import wandb

from pyagents.agents import DQNAgent, VPG, A2C
from pyagents.networks import QNetwork, PolicyNetwork, ValueNetwork, ActorCriticNetwork
from pyagents.memory import PrioritizedBuffer
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import gin


@gin.configurable
def train_dqn_agent(n_episodes=1000, batch_size=64, learning_rate=0.001, steps_to_train=5,
                    max_steps=500, min_memories=5000, output_dir="./output/", wandb_params=None):

    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    buffer = PrioritizedBuffer()
    q_net = QNetwork(state_size, action_size)
    optim = Adam(learning_rate=learning_rate)
    player = DQNAgent(state_size, action_size, q_network=q_net, buffer=buffer, optimizer=optim,
                      name='cartpole', wandb_params=wandb_params,
                      log_dict={'learning_rate': learning_rate})
    # player = DQNAgent.load('output/cartpole', ver=5, epsilon=0.01, training=False)
    if player.is_logging:
        wandb.define_metric('score', step_metric="episode", summary="max")
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
        eps_history.append(player.policy.epsilon)
        if episode % 10 == 0:
            print(f'EPISODE: {episode:4d}/{n_episodes:4d}, '
                  f'SCORE: {this_episode_score:3.0f}, '
                  f'EPS: {player.policy.epsilon:.2f}')

        if player.is_logging:
            wandb.log({'score': scores[-1], 'episode': episode, 'eps': player.policy.epsilon})

        player.train(batch_size)

        if (episode % 100) == 0:
            player.save(ver=episode//100)

    return scores, movavg100, eps_history


@gin.configurable
def train_vpg_agent(gamma, n_episodes=1000, max_steps=250,
                    actor_learning_rate=1e-4, critic_learning_rate=1e-3, schedule=True,
                    output_dir="./output/", wandb_params=None):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = (env.action_space.n, )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    a_net = PolicyNetwork(state_size, action_size, distribution='softmax', fc_layer_params=(16, 16), dropout_params=(0.1, 0.1))
    v_net = ValueNetwork(state_size, fc_layer_params=(16, 16), dropout_params=(0.1, 0.1))
    if schedule:
        a_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_learning_rate, n_episodes, actor_learning_rate/100)
        c_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_learning_rate, n_episodes, critic_learning_rate/100)
    else:
        a_lr = actor_learning_rate
        c_lr = critic_learning_rate
    a_opt = Adam(learning_rate=a_lr)
    v_opt = Adam(learning_rate=c_lr)
    player = VPG(state_size, action_size,
                 actor=a_net, critic=v_net, actor_opt=a_opt, critic_opt=v_opt,
                 gamma=gamma, name='pendulum', wandb_params=wandb_params,
                 log_dict={'actor_learning_rate': actor_learning_rate, 'critic_learning_rate': critic_learning_rate})
    if player.is_logging:
        wandb.define_metric('score', step_metric="episode", summary="max")
    scores = []
    movavg100 = []

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
            r_t = r_t if not done else -10
            s_tp1 = np.reshape(s_tp1, player.state_shape)
            player.remember(state=s_t, action=a_t, reward=r_t)
            s_t = s_tp1
            step += 1
            score += r_t

        losses = player.train()
        if player.is_logging:
            wandb.log({'score': step, 'episode': episode})
        scores.append(step)
        this_episode_score = np.mean(scores[-100:])
        movavg100.append(this_episode_score)

        if episode % 1 == 0:
            print(f'EPISODE: {episode:4d}/{n_episodes:4d}, SCORE: {this_episode_score:3.0f}, '
                  f'POLICY LOSS: {losses["policy_loss"]:4.2f}, CRITIC LOSS: {losses["critic_loss"]:5.2f}')

        # if (episode % 100) == 0:
        #     player.save(ver=episode//100)

    return scores, movavg100


@gin.configurable
def train_a2c_agent(gamma, n_episodes=1000, max_steps=250,
                    learning_rate=1e-3, schedule=True,
                    output_dir="./output/", wandb_params=None):
    env = gym.make('CartPole-v1')
    state_shape = env.observation_space.shape[0]
    action_shape = (env.action_space.n, )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ac_net = ActorCriticNetwork(state_shape, action_shape, distribution='softmax')
    if schedule:
        lr = tf.keras.optimizers.schedules.PolynomialDecay(learning_rate, n_episodes*10, learning_rate/100)
    else:
        lr = learning_rate
    opt = Adam(learning_rate=lr)
    player = A2C(state_shape, action_shape, actor_critic=ac_net, opt=opt, gamma=gamma,
                 name='cartpole', wandb_params=wandb_params,
                 log_dict={'actor_learning_rate': learning_rate, 'critic_learning_rate': learning_rate})
    # player = DQNAgent.load('output/cartpole', ver=5, epsilon=0.01, training=False)
    if player.is_logging:
        wandb.define_metric('score', step_metric="episode", summary="max")
    scores = []
    movavg100 = []

    for episode in range(1, n_episodes+1):

        s_t = env.reset()
        s_t = np.reshape(s_t, player.state_shape)
        step = 0
        score = 0
        done = False
        losses = None

        while not done and step < max_steps:
            # env.render()
            a_t = player.act(s_t)
            s_tp1, r_t, done, info = env.step(a_t)
            r_t = r_t if not done else -(200 - step)
            s_tp1 = np.reshape(s_tp1, player.state_shape)
            player.remember(state=s_t, action=a_t, reward=r_t)
            s_t = s_tp1
            step += 1
            score += r_t
            loss_info = player.train(done=done, next_state=s_tp1)
            losses = loss_info if loss_info is not None else losses

        scores.append(step)
        this_episode_score = np.mean(scores[-100:])
        movavg100.append(this_episode_score)
        if player.is_logging:
            wandb.log({'score': step, 'episode': episode})
        if episode % 1 == 0:
            if losses is not None:
                loss_str = f', POLICY LOSS: {losses["policy_loss"]:4.2f}, CRITIC LOSS: {losses["critic_loss"]:5.2f}'
            else:
                loss_str = ''
            print(f'EPISODE: {episode:4d}/{n_episodes:4d}, SCORE: {this_episode_score:3.0f}' + loss_str)

        # if (episode % 100) == 0:
        #     player.save(ver=episode//100)

    return scores, movavg100


def reset_random_seed(seed):
   os.environ['PYTHONHASHSEED'] = str(seed)
   tf.random.set_seed(seed)
   np.random.seed(seed)
   random.seed(seed)


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training a sample agent on Gym")
    parser.add_argument('-a', '--agent', type=str, help='which agent to train, either DQN, VPG or A2C ')
    parser.add_argument('-c', '--config', type=str, help='path to gin config file', default='')
    args = parser.parse_args()
    if args.config:
        gin.parse_config_file(args.config)
    if args.agent == 'dqn':
        info = train_dqn_agent()
    elif args.agent == 'vpg':
        info = train_vpg_agent()
    elif args.agent == 'a2c':
        info = train_a2c_agent()
    else:
        raise ValueError(f'wrong agent {args.agent}')
