import os
import random
from collections import defaultdict

import gymnasium as gym
import numpy as np
import tensorflow as tf
import gin
import wandb
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

import pyagents.agents as agents
import pyagents.networks as networks


@gin.configurable
def get_optimizer(learning_rate=0.001):
    return Adam(learning_rate=learning_rate)


@gin.configurable
def get_agent(algo, env, output_dir, act_start_learning_rate=3e-4, buffer='uniform',
              crit_start_learning_rate=None, alpha_start_learning_rate=None, schedule=True, wandb_params=None, gym_id=None, training_steps=10 ** 5,
              log_dict=None):
    if log_dict is None:
        log_dict = dict()
    log_dict.update({'schedule': schedule})

    if crit_start_learning_rate is None:
        crit_start_learning_rate = act_start_learning_rate
    if alpha_start_learning_rate is None:
        alpha_start_learning_rate = act_start_learning_rate
    if schedule:
        act_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(act_start_learning_rate, training_steps, 0.)
        crit_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(crit_start_learning_rate, training_steps, 0.)
        alpha_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(alpha_start_learning_rate, training_steps, 0.)
    else:
        act_learning_rate = act_start_learning_rate
        crit_learning_rate = crit_start_learning_rate
        alpha_learning_rate = alpha_start_learning_rate

    if isinstance(env, gym.vector.VectorEnv):
        state_shape = env.single_observation_space.shape
        action_space = env.single_action_space
    else:
        state_shape = env.observation_space.shape
        action_space = env.action_space

    if wandb_params is not None:
        wandb_params['group'] = gym_id

    if isinstance(action_space, gym.spaces.Discrete):
        action_shape = action_space.n
        output = 'softmax'
        bounds = None
    else:
        action_shape = action_space.shape
        output = 'gaussian'
        bounds = (action_space.low, action_space.high)

    if algo in ('vpg', 'ppo'):
        if isinstance(action_space, gym.spaces.Discrete):
            action_shape = (action_shape, )
        a_net = networks.PolicyNetwork(state_shape, action_shape, output=output, bounds=bounds)
        v_net = networks.ValueNetwork(state_shape)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        v_opt = get_optimizer(learning_rate=crit_learning_rate)
        log_dict['actor_learning_rate'] = act_start_learning_rate
        log_dict['critic_learning_rate'] = crit_start_learning_rate
        if algo == 'vpg':
            agent = agents.VPG(state_shape, action_shape,
                               actor=a_net, critic=v_net, actor_opt=a_opt, critic_opt=v_opt,
                               name='vpg', wandb_params=wandb_params, save_dir=output_dir,
                               log_dict=log_dict)
        else:
            agent = agents.PPO(state_shape, action_shape,
                               actor=a_net, critic=v_net, actor_opt=a_opt, critic_opt=v_opt,
                               name='ppo', wandb_params=wandb_params, save_dir=output_dir,
                               log_dict=log_dict)
    elif algo == 'a2c':
        ac_net = networks.SharedBackboneACNetwork(state_shape, action_shape, output=output, bounds=bounds)
        opt = get_optimizer(learning_rate=act_learning_rate)
        log_dict['learning_rate'] = act_start_learning_rate
        agent = agents.A2C(state_shape, action_shape, actor_critic=ac_net, opt=opt,
                           name='a2c', wandb_params=wandb_params, save_dir=output_dir,
                           log_dict=log_dict)
    elif algo == 'dqn':
        assert isinstance(action_space, gym.spaces.Discrete), 'DQN only works in discrete environments'
        action_shape = action_space.n
        log_dict['learning_rate'] = act_start_learning_rate
        q_net = networks.DiscreteQNetwork(state_shape, action_shape)
        optim = Adam(learning_rate=act_learning_rate)
        agent = agents.DQNAgent(state_shape, action_shape, q_network=q_net, buffer=buffer, optimizer=optim,
                                name='dqn', wandb_params=wandb_params, save_dir=output_dir,
                                log_dict=log_dict)
    elif algo == 'c51':
        assert isinstance(action_space, gym.spaces.Discrete), 'DQN only works in discrete environments'
        action_shape = action_space.n
        log_dict['learning_rate'] = act_start_learning_rate
        q_net = networks.C51QNetwork(state_shape, action_shape)
        optim = Adam(learning_rate=act_learning_rate)
        agent = agents.C51DQNAgent(state_shape, action_shape, q_net, buffer=buffer, optimizer=optim,
                                   name='c51', wandb_params=wandb_params, save_dir=output_dir,
                                   log_dict=log_dict)
    elif algo == 'qrdqn':
        assert isinstance(action_space, gym.spaces.Discrete), 'DQN only works in discrete environments'
        action_shape = action_space.n
        log_dict['learning_rate'] = act_start_learning_rate
        q_net = networks.QRQNetwork(state_shape, action_shape)
        optim = Adam(learning_rate=act_learning_rate, epsilon=0.01 / 32)
        agent = agents.QRDQNAgent(state_shape, action_shape, q_net, buffer=buffer, optimizer=optim,
                                  name='qrdqn', wandb_params=wandb_params, save_dir=output_dir,
                                  log_dict=log_dict)

    elif algo == 'ddpg':
        assert isinstance(action_space, gym.spaces.Box), 'DDPG only works in continuous spaces'
        pi_params = {'bounds': bounds,
                     'out_params': {'activation': 'tanh'}}
        q_params = {}
        ac = networks.ACNetwork(state_shape=state_shape, action_shape=action_shape,
                                pi_out='continuous', pi_params=pi_params, q_params=q_params)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        c_opt = get_optimizer(learning_rate=crit_learning_rate)
        log_dict['actor_learning_rate'] = act_start_learning_rate
        log_dict['critic_learning_rate'] = crit_start_learning_rate
        agent = agents.DDPG(state_shape, action_shape, actor_critic=ac, buffer=buffer,
                            actor_opt=a_opt, critic_opt=c_opt,
                            action_bounds=bounds, name='ddpg', wandb_params=wandb_params, save_dir=output_dir,
                            log_dict=log_dict)
    elif algo == 'sac':
        assert isinstance(action_space, gym.spaces.Box), 'sac only works in continuous spaces'
        action_shape = action_space.shape
        bounds = (action_space.low, action_space.high)

        a_net = networks.PolicyNetwork(state_shape, action_shape, output='gaussian', bounds=bounds,
                                       activation='relu', out_params={'state_dependent_std': True,
                                                                      'mean_activation': None})
        q1_net = networks.QNetwork(state_shape=state_shape, action_shape=action_shape)
        q2_net = networks.QNetwork(state_shape=state_shape, action_shape=action_shape)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        c1_opt = get_optimizer(learning_rate=crit_learning_rate)
        c2_opt = get_optimizer(learning_rate=crit_learning_rate)
        alpha_opt = get_optimizer(learning_rate=alpha_learning_rate)

        agent = agents.SAC(state_shape, action_shape, actor=a_net, buffer=buffer,
                           critic=q1_net, actor_opt=a_opt, critic1_opt=c1_opt,
                           critic2=q2_net, critic2_opt=c2_opt,
                           alpha_opt=alpha_opt, wandb_params=wandb_params, save_dir=output_dir,
                           log_dict={'actor_learning_rate': act_start_learning_rate,
                                     'critic_learning_rate': crit_start_learning_rate,
                                     'alpha_learning_rate': alpha_start_learning_rate})
    else:
        raise ValueError(f'unsupported algorithm {algo}')
    return agent


def train_on_policy_agent(batch_size=128, rollout_steps=100, update_rounds=1):
    def train_step(agent, envs, s_t):
        train_info = defaultdict(lambda: list())
        for _ in range(rollout_steps):
            agent_out = agent.act(s_t)
            a_t, lp_t = agent_out.actions, agent_out.logprobs
            s_tp1, r_t, terminated, truncated, info = envs.step(a_t)
            if 'final_info' in info:
                for single_step in filter(lambda x: x is not None, info['final_info']):
                    train_info['avg_ret'].append(single_step['episode']['r'])
                    train_info['avg_len'].append(single_step['episode']['l'])

            agent.remember(state=s_t,
                           action=a_t,
                           reward=r_t,
                           next_state=s_tp1,
                           done=terminated,
                           logprob=lp_t)
            s_t = s_tp1

        loss_dict = agent.train(batch_size, update_rounds=update_rounds)
        train_info['train_step'] = agent.train_step
        train_info = {**train_info, **loss_dict}
        return s_t, train_info

    return train_step


def train_off_policy_agent(batch_size=128, rollout_steps=100, update_rounds=1):
    def train_step(agent, envs, s_t):
        train_info = defaultdict(lambda: list())
        for _ in range(rollout_steps):
            agent_out = agent.act(s_t)
            a_t, lp_t = agent_out.actions, agent_out.logprobs
            s_tp1, r_t, terminated, truncated, info = envs.step(a_t)
            if 'final_info' in info:
                for single_step in filter(lambda x: x is not None, info['final_info']):
                    train_info['avg_ret'].append(single_step['episode']['r'])
                    train_info['avg_len'].append(single_step['episode']['l'])

            agent.remember(state=s_t,
                           action=a_t,
                           reward=r_t,
                           next_state=s_tp1,
                           done=terminated,
                           logprob=lp_t)
            s_t = s_tp1

        loss_dict = defaultdict(lambda: 0)  # keeps track of average losses
        for _ in range(update_rounds):
            epoch_info = agent.train(batch_size)
            for loss, value in epoch_info.items():
                loss_dict[loss] += (value / update_rounds)

        train_info['train_step'] = agent.train_step
        train_info = {**train_info, **loss_dict}
        return s_t, train_info

    return train_step


@gin.configurable
def train_agent(agent, train_envs, test_env=None, train_step_fn=None, training_steps=10 ** 5, batch_size=64,
                update_rounds=1,
                rollout_steps=100, init_params=None, output_dir="./output/", test_every=100, seed=42, test_rounds=100,
                unique_seed=False):
    """Performs training of the agent on given train_envs for training_steps, optionally testing the agents
       every test_rounds if test_env is provided. unique_seed forces the same seed(s) for training and testing."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if init_params is not None:
        assert isinstance(init_params, dict), f'init_params should be dict of optional parameters for init function'
    else:
        init_params = {}

    training_step = 0
    best_score = float('-inf')
    ver = 1
    episodes = 0
    scores = test_agent(agent, test_env, seed=seed, n_episodes=test_rounds)
    if agent.is_logging:
        wandb.log({'train_step': 0, 'test/score': np.mean(scores)})
    info = dict(avg_ret=np.mean(scores), avg_len=0)
    print(f'{"*" * 42}\nSTARTING TRAINING\n{"*" * 42}')
    with tqdm(total=training_steps) as pbar:
        pbar.set_description('INITIALIZING')
        env_config = dict(batch_size=batch_size, rollout_steps=rollout_steps, num_envs=train_envs.num_envs,
                          update_rounds=update_rounds, training_steps=training_steps)

        if agent.on_policy:
            if train_step_fn is None:
                train_step_fn = train_on_policy_agent(batch_size=batch_size,
                                                      rollout_steps=rollout_steps,
                                                      update_rounds=update_rounds)
            agent.init(train_envs, rollout_steps, env_config=env_config, **init_params)
        else:
            if train_step_fn is None:
                train_step_fn = train_off_policy_agent(batch_size=batch_size,
                                                       rollout_steps=rollout_steps,
                                                       update_rounds=update_rounds)
            agent.init(train_envs, env_config=env_config, **init_params)

        state, new_info = train_envs.reset() #seed=seed
        info.update(new_info)

        if not unique_seed:
            seed = ((seed ** 2) + 33) // 2  # generate new random seed for testing, pretty arbitrary here
        pbar.set_description('TRAINING')
        while training_step <= training_steps:
            state, new_info = train_step_fn(agent, train_envs, s_t=state)
            training_step += update_rounds
            info.update(new_info)

            if test_env is not None and training_step > ver * test_every:
                pbar.set_description('TESTING')
                scores = test_agent(agent, test_env, seed=seed, n_episodes=test_rounds)
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    agent.save(ver=ver)
                ver += 1
                info['test/score'] = avg_score
                pbar.set_description(f'[EVAL SCORE: {avg_score:4.0f}] TRAINING')

            if agent.is_logging:
                info['avg_ret'] = np.mean(info['avg_ret'])
                info['avg_len'] = np.mean(info['avg_len'])
                wandb.log(info)

            pbar.update(update_rounds)
            pbar.set_postfix(**info)

    agent.save(ver=0)
    return agent, scores


def test_agent(agent, envs, seed, n_episodes):
    def no_vec_test(env, s_t):
        score, episode = 0, 0
        s_t = tf.expand_dims(s_t, axis=0)
        a_t = agent.act(s_t, training=False).actions[0]
        s_tp1, _, terminated, truncated, info = env.step(a_t)
        if "episode" in info.keys():
            score = [info['episode']['r']]
            episode = 1
        if terminated:
            s_tp1, info = envs.reset()
        return score, episode, s_tp1

    def vec_test(envs, s_t):
        episodes = 0
        scores = []
        a_t = agent.act(s_t, training=False).actions
        s_tp1, _, _, _, info = envs.step(a_t)
        if 'final_info' in info:
            for single_step in filter(lambda x: x is not None, info['final_info']):
                scores.append(single_step['episode']['r'])
                episodes += 1
        return scores, episodes, s_tp1

    scores = []
    episode = 0
    s_t, info = envs.reset(seed=seed)
    test_step_fn = vec_test if isinstance(envs, gym.vector.VectorEnv) else no_vec_test
    while episode < n_episodes:
        step_scores, step_episodes, s_t = test_step_fn(envs, s_t)
        if step_episodes > 0:
            episode += step_episodes
            scores += step_scores
    return np.array(scores)


def load_agent(algo, path, ver):
    if algo == 'dqn':
        agent = agents.DQNAgent.load(path, ver=ver, epsilon=0.00, training=False)
    elif algo == 'c51':
        agent = agents.C51DQNAgent.load(path, ver=ver, training=False)
    elif algo == 'vpg':
        agent = agents.VPG.load(path, ver=ver, training=False)
    elif algo == 'a2c':
        agent = agents.A2C.load(path, ver=ver, training=False)
    elif algo == 'ddpg':
        agent = agents.DDPG.load(path, ver=ver, training=False)
    elif algo == 'ppo':
        agent = agents.PPO.load(path, ver=ver, training=False)
    elif algo == 'qrdqn':
        agent = agents.QRDQNAgent.load(path, ver=ver, training=False)
    elif algo == 'sac':
        agent = agents.SAC.load(path, ver=ver, training=False)
    else:
        raise ValueError(f'unsupported algorithm {algo}')
    return agent


@gin.configurable
def get_envs(n_envs, gym_id, seed, capture_video, output_dir, frame_stack=1, async_envs=False, no_vect=False,
             record_every=10, rew_fn=lambda r: r):
    """Creates vectorized environments."""

    def make_env(gym_id, seed, idx, capture_video, output_dir):
        """Auxiliary functions used by gym.vector to create the individual env."""

        def thunk():
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            env = gym.make(gym_id, render_mode='human' if capture_video else None)

            if capture_video and idx == 0:
                if not os.path.isdir(f"{output_dir}/videos"):
                    os.mkdir(f"{output_dir}/videos")
                env = gym.wrappers.RecordVideo(env, f"{output_dir}/videos",
                                               episode_trigger=lambda e: (e % record_every) == 5)

            if frame_stack > 1:
                env = gym.wrappers.FrameStack(env, num_stack=frame_stack)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return thunk
    if no_vect:
        envs = make_env(gym_id, seed, 0, capture_video, output_dir)()
    else:
        if async_envs:
            vec_fn = gym.vector.AsyncVectorEnv
        else:
            vec_fn = gym.vector.SyncVectorEnv
        envs = vec_fn(
            [make_env(gym_id, seed + i, i, capture_video if i == 0 else False, output_dir)
             for i in range(n_envs)])

    return envs


def reset_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
