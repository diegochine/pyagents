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
    if isinstance(env, gym.vector.VectorEnv):
        state_shape = env.single_observation_space.shape
        action_space = env.single_action_space
    else:
        state_shape = env.observation_space.shape
        action_space = env.action_space
    if algo == 'dqn':
        assert isinstance(action_space, gym.spaces.Discrete), 'DQN only works in discrete environments'
        action_shape = action_space.n
        buffer = PrioritizedBuffer() if buffer == 'prioritized' else UniformBuffer()
        q_net = DiscreteQNetwork(state_shape, action_shape)
        optim = Adam(learning_rate=act_learning_rate)
        agent = DQNAgent(state_shape, action_shape, q_network=q_net, buffer=buffer, optimizer=optim,
                         name='dqn', wandb_params=wandb_params, save_dir=output_dir,
                         log_dict={'learning_rate': act_start_learning_rate})
    elif algo == 'vpg':
        if isinstance(action_space, gym.spaces.Discrete):
            action_shape = (action_space.n,)
            output = 'softmax'
            bounds = None
        else:
            action_shape = action_space.shape
            output = 'gaussian'
            bounds = (action_space.low, action_space.high)
        a_net = PolicyNetwork(state_shape, action_shape, output=output, bounds=bounds, scaling=scaling)
        v_net = ValueNetwork(state_shape)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        v_opt = get_optimizer(learning_rate=crit_learning_rate)
        agent = VPG(state_shape, action_shape,
                    actor=a_net, critic=v_net, actor_opt=a_opt, critic_opt=v_opt,
                    name='vpg', wandb_params=wandb_params, save_dir=output_dir,
                    log_dict={'actor_learning_rate': act_start_learning_rate,
                              'critic_learning_rate': crit_start_learning_rate})
    elif algo == 'a2c':
        if isinstance(action_space, gym.spaces.Discrete):
            action_shape = (action_space.n,)
            output = 'softmax'
            bounds = None
        else:
            action_shape = action_space.shape
            output = 'beta'
            bounds = (action_space.low, action_space.high)
        ac_net = SharedBackboneACNetwork(state_shape, action_shape, output=output, bounds=bounds)
        opt = get_optimizer(learning_rate=act_learning_rate)
        agent = A2C(state_shape, action_shape, actor_critic=ac_net, opt=opt,
                    name='a2c', wandb_params=wandb_params, save_dir=output_dir,
                    log_dict={'learning_rate': act_start_learning_rate})
    elif algo == 'ddpg':
        assert isinstance(action_space, gym.spaces.Box), 'DDPG only works in continuous spaces'
        action_shape = action_space.shape
        bounds = (action_space.low, action_space.high)
        buffer = PrioritizedBuffer() if buffer == 'prioritized' else UniformBuffer()
        pi_params = {'bounds': bounds, 'scaling': scaling,
                     'out_params': {'activation': 'tanh'}}
        q_params = {}
        ac = ACNetwork(state_shape=state_shape, action_shape=action_shape,
                       pi_out='continuous', pi_params=pi_params, q_params=q_params)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        c_opt = get_optimizer(learning_rate=crit_learning_rate)
        agent = DDPG(state_shape, action_shape, actor_critic=ac, buffer=buffer, actor_opt=a_opt, critic_opt=c_opt,
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


def make_env(gym_id, seed, idx, capture_video, output_dir):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"{output_dir}/videos")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training a sample agent on Gym")
    parser.add_argument('-a', '--agent', type=str, help='which agent to use, either DQN, VPG, A2C or DDPG')
    parser.add_argument('-c', '--config', type=str, default='', help='path to gin config file')
    parser.add_argument('-tv', '--test-ver', type=int, default=-1,
                        help='if -1, trains; if >=0, performs evaluation using this version')
    parser.add_argument('-e', '--env', type=str, default='cartpole',
                        help='env to use, choices: cartpole, pendulum, bipedalwalker')
    parser.add_argument('-n', '--num-envs', type=int, default=1,
                        help='number of parallel envs for vectorized environment')
    args = parser.parse_args()
    args.env = args.env.lower()
    if args.env.startswith('c'):
        gym_id = 'CartPole-v1'
    elif args.env.startswith('p'):
        gym_id = 'Pendulum-v1'
    elif args.env.startswith('b'):
        gym_id = 'BipedalWalker-v3'
    else:
        raise ValueError(f'unsupported env {args.env}')
    output_dir = f'output/{args.agent}-{gym_id}'
    seed = 42
    # env = gym.wrappers.NormalizeObservation(env)
    if args.config:
        gin.parse_config_file(args.config)
    is_testing = (args.test_ver >= 0)
    if is_testing:
        agent = load_agent(args.agent, output_dir, args.test_ver)
        env = make_env(gym_id, seed, 0, True, output_dir)()
        scores = test_agent(agent, env)
    else:
        envs = gym.vector.SyncVectorEnv([make_env(gym_id, seed, i, False, output_dir) for i in range(args.num_envs)])
        agent = get_agent(args.agent, envs, output_dir)
        agent, scores = train_agent(agent, envs, output_dir=output_dir)
        agent.save(f'output/{args.agent}-{args.env}', ver='final')
