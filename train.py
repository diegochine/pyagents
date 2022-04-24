import os
import random

import numpy as np
import gym
from argparse import ArgumentParser

from pyagents.agents import DQNAgent, VPG, A2C, DDPG
from pyagents.networks import DiscreteQNetwork, PolicyNetwork, ValueNetwork, SharedBackboneACNetwork, ACNetwork
from pyagents.memory import PrioritizedBuffer, UniformBuffer
from pyagents.utils import train_agent
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import gin

from pyagents.utils.training_utils import test_agent, get_optimizer


def load_agent(algo, dir, ver):
    if algo == 'dqn':
        agent = DQNAgent.load(dir, ver=ver, epsilon=0.00, training=False)
    elif algo == 'vpg':
        agent = VPG.load(dir, ver=ver, training=False)
    elif algo == 'a2c':
        agent = A2C.load(dir, ver=ver, training=False)
    elif algo == 'ddpg':
        agent = DDPG.load(dir, ver=ver, training=False)
    else:
        raise ValueError(f'unsupported algorithm {algo}')
    return agent


@gin.configurable
def get_agent(algo, env, output_dir, act_start_learning_rate=1e-3, buffer='uniform', scaling=None,
              crit_start_learning_rate=None, schedule=False, wandb_params=None):
    if crit_start_learning_rate is None:
        crit_start_learning_rate = act_start_learning_rate
    if schedule:
        act_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(act_start_learning_rate, 10 ** 5,
                                                                          act_start_learning_rate / 100)
        crit_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(crit_start_learning_rate, 10 ** 5,
                                                                           crit_start_learning_rate / 100)
    else:
        act_learning_rate = act_start_learning_rate
        crit_learning_rate = crit_start_learning_rate
    state_shape = env.observation_space.shape
    if algo == 'dqn':
        action_size = env.action_space.n
        buffer = PrioritizedBuffer() if buffer == 'prioritized' else UniformBuffer()
        q_net = DiscreteQNetwork(state_shape, action_size)
        optim = Adam(learning_rate=act_learning_rate)
        agent = DQNAgent(state_shape, action_size, q_network=q_net, buffer=buffer, optimizer=optim,
                         name='dqn', wandb_params=wandb_params, save_dir=output_dir,
                         log_dict={'learning_rate': act_start_learning_rate})

    elif algo == 'vpg':
        action_size = (env.action_space.n,)
        a_net = PolicyNetwork(state_shape, action_size, output='softmax')
        v_net = ValueNetwork(state_shape)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        v_opt = get_optimizer(learning_rate=crit_learning_rate)
        agent = VPG(state_shape, action_size,
                    actor=a_net, critic=v_net, actor_opt=a_opt, critic_opt=v_opt,
                    name='vpg', wandb_params=wandb_params, save_dir=output_dir,
                    log_dict={'actor_learning_rate': act_start_learning_rate,
                              'critic_learning_rate': crit_start_learning_rate})
    elif algo == 'a2c':
        action_shape = (env.action_space.n,)
        ac_net = SharedBackboneACNetwork(state_shape, action_shape, output='softmax')
        opt = get_optimizer(learning_rate=act_learning_rate)
        agent = A2C(state_shape, action_shape, actor_critic=ac_net, opt=opt,
                    name='a2c', wandb_params=wandb_params, save_dir=output_dir,
                    log_dict={'learning_rate': act_start_learning_rate})

    elif algo == 'ddpg':
        action_size = env.action_space.shape
        bounds = (env.action_space.low, env.action_space.high)
        buffer = PrioritizedBuffer() if buffer == 'prioritized' else UniformBuffer()
        pi_params = {'bounds': bounds, 'scaling': scaling,
                     'out_params': {'activation': 'tanh'}}
        q_params = {}
        ac = ACNetwork(state_shape=state_shape, action_shape=action_size,
                       pi_out='continuous', pi_params=pi_params, q_params=q_params)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        c_opt = get_optimizer(learning_rate=crit_learning_rate)
        agent = DDPG(state_shape, action_size, actor_critic=ac, buffer=buffer, actor_opt=a_opt, critic_opt=c_opt,
                     action_bounds=bounds, name='ddpg', wandb_params=wandb_params, save_dir=output_dir,
                     log_dict={'actor_learning_rate': act_start_learning_rate,
                               'critic_learning_rate': crit_start_learning_rate})
    else:
        raise ValueError(f'unsupported algorithm {algo}')
    return agent


def reset_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training a sample agent on Gym")
    parser.add_argument('-a', '--agent', type=str, help='which agent to use, either DQN, VPG, A2C or DDPG')
    parser.add_argument('-c', '--config', type=str, default='', help='path to gin config file')
    parser.add_argument('-tv', '--test-ver', type=int, default=-1,
                        help='if -1, trains; if >=0, performs evaluation using this version')
    parser.add_argument('-e', '--env', type=str, default='cartpole',
                        help='env to use, choices: cartpole, pendulum, bipedalwalker')
    args = parser.parse_args()
    args.env = args.env.lower()
    if args.env.startswith('c'):
        env = gym.make('CartPole-v1')
    elif args.env.startswith('p'):
        env = gym.make('Pendulum-v1')
    elif args.env.startswith('b'):
        env = gym.make('BipedalWalker-v3')
    else:
        raise ValueError(f'unsupported env {args.env}')
    # env = gym.wrappers.NormalizeObservation(env)
    if args.config:
        gin.parse_config_file(args.config)
    output_dir = f'output/{args.agent}-{args.env}'
    is_testing = (args.test_ver >= 0)
    if is_testing:
        agent = load_agent(args.agent, output_dir, args.test_ver)
        scores = test_agent(agent, env)
    else:
        agent = get_agent(args.agent, env, output_dir)
        agent, scores = train_agent(agent, env, output_dir=output_dir)
        agent.save(f'output/{args.agent}-{args.env}', ver='final')
