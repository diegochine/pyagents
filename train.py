import argparse
import os

import numpy as np
from argparse import ArgumentParser

from pyagents.utils import train_agent
import gin

from pyagents.utils.training_utils import test_agent, get_agent, load_agent, get_envs

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
                        help='random seed; each training env is seeded with (seed + i) where i is the individual'
                             'env index')
    parser.add_argument('--test-envs', type=int, default=5,
                        help='number of testing environments')
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
    elif args.env.startswith('wa'):
        gym_id = 'Walker2d-v2'
    elif args.env.startswith('ha'):
        gym_id = 'HalfCheetah-v3'
    elif args.env.startswith('viz'):
        import vizdoom.gym_wrapper
        gym_id = 'VizdoomDefendLine-v0'
        # ['VizdoomBasic-v0', 'VizdoomCorridor-v0', 'VizdoomDefendCenter-v0', 'VizdoomDefendLine-v0',
        # 'VizdoomHealthGathering-v0', 'VizdoomMyWayHome-v0', 'VizdoomPredictPosition-v0', 'VizdoomTakeCover-v0',
        # 'VizdoomDeathmatch-v0', 'VizdoomHealthGatheringSupreme-v0']
    else:
        raise ValueError(f'unsupported env {args.env}')

    if not args.output_dir:
        args.output_dir = f'output/{args.agent}-{gym_id.replace("/", "-")}'

    if args.test_ver is not None:
        agent = load_agent(args.agent, args.output_dir, args.test_ver)
        envs = get_envs(n_envs=args.test_envs, seed=args.seed, gym_id=gym_id,
                        capture_video=args.video, output_dir=args.output_dir, no_vect=True)
        scores = test_agent(agent, envs, render=True, seed=args.seed, n_episodes=100)
        avg_score = np.mean(scores)
        print(f'AVG SCORE: {avg_score:4.0f}')
        exit()

    listdir = sorted(os.listdir(args.config_dir))
    for cfg_file in listdir:
        gin.parse_config_file(os.path.join(args.config_dir, cfg_file))
        out_dir = f'{args.output_dir}_{cfg_file.split(".")[0]}'
        train_envs = get_envs(n_envs=args.num_envs, seed=args.seed, gym_id=gym_id,
                              capture_video=args.video, output_dir=out_dir)
        test_envs = get_envs(n_envs=args.test_envs, seed=args.seed, gym_id=gym_id,
                             capture_video=args.video, output_dir=out_dir)
        agent = get_agent(args.agent, train_envs, output_dir=out_dir, gym_id=gym_id)
        agent, scores = train_agent(agent, train_envs, test_envs, seed=args.seed, output_dir=out_dir)
