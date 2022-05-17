import argparse
import os

import numpy as np
import gym
from argparse import ArgumentParser

from pyagents.utils import train_agent
import gin

from pyagents.utils.training_utils import test_agent, get_agent, load_agent


def make_env(gym_id, seed, idx, capture_video, output_dir):
    from pyagents.utils.training_utils import reset_random_seed
    reset_random_seed(seed)

    def thunk():
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        env_args = dict()
        if gym_id.startswith('ALE'):
            env_args = dict(full_action_space=False,  # reduced action space for easier learning
                            )
        env = gym.make(gym_id, **env_args)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video and idx == 0:
            if not os.path.isdir(f"{output_dir}/videos"):
                os.mkdir(f"{output_dir}/videos")
            env = gym.wrappers.RecordVideo(env, f"{output_dir}/videos", episode_trigger=lambda s: (s % 2) == 0)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    parser = ArgumentParser(description="Script for training a sample agent on Gym")
    parser.add_argument('-a', '--agent', type=str, help='which agent to use, either DQN, VPG, A2C or DDPG')
    parser.add_argument('-c', '--config-dir', type=str, default='', help='path to dir containing gin config file')
    parser.add_argument('-tv', '--test-ver', type=int, default=None,
                        help='if -1, use final version; if >=0, performs evaluation using this version')
    parser.add_argument('-e', '--env', type=str, default='cartpole',
                        help='env to use, choices: cartpole, pendulum, bipedalwalker')
    parser.add_argument('-n', '--num-envs', type=int, default=1,
                        help='number of parallel envs for vectorized environment')
    parser.add_argument('-o', '--output-dir', type=str, default='',
                        help='directory where to store trained agent(s)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--video', action=argparse.BooleanOptionalAction, default=False,
                        help='number of parallel envs for vectorized environment')
    args = parser.parse_args()
    args.env = args.env.lower()
    if args.env.startswith('a'):
        gym_id = 'Acrobot-v1'
    elif args.env.startswith('bi'):
        gym_id = 'BipedalWalker-v3'
    elif args.env.startswith('br'):
        gym_id = 'ALE/Breakout-ram'
    elif args.env.startswith('c'):
        gym_id = 'CartPole-v1'
    elif args.env.startswith('l'):
        gym_id = 'LunarLander-v2'
    elif args.env.startswith('m'):
        gym_id = 'MountainCar-v0'
    elif args.env.startswith('p'):
        gym_id = 'Pendulum-v1'
    else:
        raise ValueError(f'unsupported env {args.env}')

    if not args.output_dir:
        args.output_dir = f'output/{args.agent}-{gym_id.replace("/", "-")}'

    if args.test_ver is not None:
        agent = load_agent(args.agent, args.output_dir, args.test_ver)
        envs = gym.vector.SyncVectorEnv(
            [make_env(gym_id, (args.seed * (3 * i)) ** 2, i, True, args.output_dir)
             for i in range(100)])  # test for 100 runs
        scores = test_agent(agent, envs, render=False)
        avg_score = np.mean(scores)
        print(f'AVG SCORE: {avg_score:4.0f}')
        exit()

    for cfg_file in os.listdir(args.config_dir):
        gin.parse_config_file(os.path.join(args.config_dir, cfg_file))
        train_envs = gym.vector.SyncVectorEnv(
            [make_env(gym_id, (args.seed * (i + 101)) ** 2, i, False, args.output_dir)
             for i in range(args.num_envs)])
        test_envs = gym.vector.SyncVectorEnv(
            [make_env(gym_id, (args.seed * (i + 8)) ** 2, i, args.video, args.output_dir)
             for i in range(100)])  # test for 100 runs
        agent = get_agent(args.agent, train_envs, output_dir=args.output_dir, gym_id=gym_id)
        agent, scores = train_agent(agent, train_envs, test_envs, output_dir=args.output_dir)
        agent.save(ver=-1)
