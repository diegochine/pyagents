import os
import random
from collections import defaultdict

import gym
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
              crit_start_learning_rate=None, schedule=True, wandb_params=None, gym_id=None, training_steps=10 ** 5):
    if crit_start_learning_rate is None:
        crit_start_learning_rate = act_start_learning_rate
    if schedule:
        act_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(act_start_learning_rate,
                                                                          training_steps,
                                                                          act_start_learning_rate / 1000)
        crit_learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(crit_start_learning_rate,
                                                                           training_steps,
                                                                           crit_start_learning_rate / 1000)
    else:
        act_learning_rate = act_start_learning_rate
        crit_learning_rate = crit_start_learning_rate

    if isinstance(env, gym.vector.VectorEnv):
        state_shape = env.single_observation_space.shape
        action_space = env.single_action_space
    else:
        state_shape = env.observation_space.shape
        action_space = env.action_space

    if wandb_params is not None:
        wandb_params['group'] = gym_id

    if algo == 'dqn':
        assert isinstance(action_space, gym.spaces.Discrete), 'DQN only works in discrete environments'
        action_shape = action_space.n
        q_net = networks.DiscreteQNetwork(state_shape, action_shape)
        optim = Adam(learning_rate=act_learning_rate)
        agent = agents.DQNAgent(state_shape, action_shape, q_network=q_net, buffer=buffer, optimizer=optim,
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
        a_net = networks.PolicyNetwork(state_shape, action_shape, output=output, bounds=bounds)
        v_net = networks.ValueNetwork(state_shape)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        v_opt = get_optimizer(learning_rate=crit_learning_rate)
        agent = agents.VPG(state_shape, action_shape,
                           actor=a_net, critic=v_net, actor_opt=a_opt, critic_opt=v_opt,
                           name='vpg', wandb_params=wandb_params, save_dir=output_dir,
                           log_dict={'actor_learning_rate': act_start_learning_rate,
                                     'critic_learning_rate': crit_start_learning_rate})
    elif algo == 'ppo':
        if isinstance(action_space, gym.spaces.Discrete):
            action_shape = (action_space.n,)
            output = 'softmax'
            bounds = None
        else:
            action_shape = action_space.shape
            output = 'gaussian'
            bounds = (action_space.low, action_space.high)
        a_net = networks.PolicyNetwork(state_shape, action_shape, output=output, bounds=bounds)
        v_net = networks.ValueNetwork(state_shape)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        v_opt = get_optimizer(learning_rate=crit_learning_rate)
        agent = agents.PPO(state_shape, action_shape,
                           actor=a_net, critic=v_net, actor_opt=a_opt, critic_opt=v_opt,
                           name='ppo', wandb_params=wandb_params, save_dir=output_dir,
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
        ac_net = networks.SharedBackboneACNetwork(state_shape, action_shape, output=output, bounds=bounds)
        opt = get_optimizer(learning_rate=act_learning_rate)
        agent = agents.A2C(state_shape, action_shape, actor_critic=ac_net, opt=opt,
                           name='a2c', wandb_params=wandb_params, save_dir=output_dir,
                           log_dict={'learning_rate': act_start_learning_rate})
    elif algo == 'ddpg':
        assert isinstance(action_space, gym.spaces.Box), 'DDPG only works in continuous spaces'
        action_shape = action_space.shape
        bounds = (action_space.low, action_space.high)
        pi_params = {'bounds': bounds,
                     'out_params': {'activation': 'tanh'}}
        q_params = {}
        ac = networks.ACNetwork(state_shape=state_shape, action_shape=action_shape,
                                pi_out='continuous', pi_params=pi_params, q_params=q_params)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        c_opt = get_optimizer(learning_rate=crit_learning_rate)
        agent = agents.DDPG(state_shape, action_shape, actor_critic=ac, buffer=buffer,
                            actor_opt=a_opt, critic_opt=c_opt,
                            action_bounds=bounds, name='ddpg', wandb_params=wandb_params, save_dir=output_dir,
                            log_dict={'actor_learning_rate': act_start_learning_rate,
                                      'critic_learning_rate': crit_start_learning_rate})
    elif algo == 'sac':
        assert isinstance(action_space, gym.spaces.Box), 'sac only works in continuous spaces'
        action_shape = action_space.shape
        bounds = (action_space.low, action_space.high)

        a_net = networks.PolicyNetwork(state_shape, action_shape, output='gaussian', bounds=bounds,
                                       out_params={'state_dependent_std': True})
        q_net = networks.QNetwork(state_shape=state_shape, action_shape=action_shape)
        a_opt = get_optimizer(learning_rate=act_learning_rate)
        c_opt = get_optimizer(learning_rate=crit_learning_rate)
        alpha_opt = get_optimizer(learning_rate=act_start_learning_rate)

        agent = agents.SAC(state_shape, action_shape, actor=a_net,
                           critic=q_net, actor_opt=a_opt, critic_opt=c_opt,
                           alpha_opt=alpha_opt, wandb_params=wandb_params, save_dir=output_dir,
                           log_dict={'actor_learning_rate': act_start_learning_rate,
                                     'critic_learning_rate': crit_start_learning_rate,
                                     'alpha_learning_rate': act_start_learning_rate})
    else:
        raise ValueError(f'unsupported algorithm {algo}')
    return agent


def train_on_policy_agent(batch_size=128, rollout_steps=100, update_rounds=1):
    def train_step(agent, envs, s_t):
        train_info = {'avg_return': [], 'avg_len': []}
        for _ in range(rollout_steps):
            agent_out = agent.act(s_t)
            a_t, lp_t = agent_out.actions, agent_out.logprobs
            s_tp1, r_t, done, info = envs.step(a_t)
            agent.remember(state=s_t,
                           action=a_t,
                           reward=r_t,
                           next_state=s_tp1,
                           done=done,
                           logprob=lp_t)
            s_t = s_tp1
            for single_step in info:
                if "episode" in single_step:
                    train_info['avg_return'].append(single_step['episode']['r'])
                    train_info['avg_len'].append(single_step['episode']['l'])

        loss_dict = agent.train(batch_size, update_rounds=update_rounds)
        train_info['train_step'] = agent.train_step
        if train_info['avg_return']:
            train_info['avg_return'] = np.mean(train_info['avg_return'])
            train_info['avg_len'] = np.mean(train_info['avg_len'])
        else:
            train_info['avg_return'] = None
            train_info['avg_len'] = None
        train_info = {**train_info, **loss_dict}
        return s_t, train_info

    return train_step


def train_off_policy_agent(batch_size=128, rollout_steps=100, update_rounds=1):
    def train_step(agent, envs, s_t):
        train_info = {'avg_return': [], 'avg_len': []}
        for _ in range(rollout_steps):
            agent_out = agent.act(s_t)
            a_t, lp_t = agent_out.actions, agent_out.logprobs
            s_tp1, r_t, done, info = envs.step(a_t)
            agent.remember(state=s_t,
                           action=a_t,
                           reward=r_t,
                           next_state=s_tp1,
                           done=done,
                           logprob=lp_t)
            s_t = s_tp1
            for single_step in info:
                if "episode" in single_step:
                    train_info['avg_return'].append(single_step['episode']['r'])
                    train_info['avg_len'].append(single_step['episode']['l'])

        loss_dict = defaultdict(lambda: 0)  # keeps track of average losses
        for epoch in range(update_rounds):
            epoch_info = agent.train(batch_size)
            for loss, value in epoch_info.items():
                loss_dict[loss] += (value / update_rounds)

        train_info['train_step'] = agent.train_step
        if train_info['avg_return']:
            train_info['avg_return'] = np.mean(train_info['avg_return'])
            train_info['avg_len'] = np.mean(train_info['avg_len'])
        else:
            train_info['avg_return'] = None
            train_info['avg_len'] = None
        train_info = {**train_info, **loss_dict}
        return s_t, train_info

    return train_step


@gin.configurable
def train_agent(agent, train_envs, test_env=None, training_steps=10 ** 5, batch_size=64, update_rounds=1,
                rollout_steps=100, init_params=None, output_dir="./output/", test_every=100, test_rounds=100):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if init_params is not None:
        assert isinstance(init_params, dict), f'init_params should be dict of optional parameters for init function'
    else:
        init_params = {}

    scores = []
    losses = []
    training_step = 0
    env_step = 0
    best_score = float('-inf')
    k = 0
    avg_r, avg_l = 0, 0

    print(f'{"*" * 42}\nSTARTING TRAINING\n{"*" * 42}')
    with tqdm(total=training_steps) as pbar:
        pbar.set_description('INITIALIZING')
        env_config = dict(batch_size=batch_size, rollout_steps=rollout_steps, num_envs=train_envs.num_envs,
                          update_rounds=update_rounds, training_steps=training_steps)
        if agent.on_policy:
            train_step_fn = train_on_policy_agent(batch_size=batch_size,
                                                  rollout_steps=rollout_steps,
                                                  update_rounds=update_rounds)
            agent.init(train_envs, rollout_steps, env_config=env_config, **init_params)
        else:
            train_step_fn = train_off_policy_agent(batch_size=batch_size,
                                                   rollout_steps=rollout_steps,
                                                   update_rounds=update_rounds)
            agent.init(train_envs, env_config=env_config, **init_params)

        state = train_envs.reset()
        pbar.set_description('TRAINING')
        while training_step <= training_steps:
            state, info = train_step_fn(agent, train_envs, s_t=state)
            training_step += update_rounds

            if test_env is not None and training_step > k * test_every:
                pbar.set_description('TESTING')
                scores = test_agent(agent, test_env, render=False)
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    agent.save(ver=k)
                k += 1
                info['test/score'] = avg_score
                pbar.set_description(f'[EVAL SCORE: {avg_score:4.0f}] TRAINING')

            if info['avg_return'] is not None:  # handle prettier tqdm printing
                avg_r = info['avg_return']
                avg_l = info['avg_len']
            else:
                info['avg_return'] = avg_r
                info['avg_len'] = avg_l

            pbar.update(update_rounds)
            pbar.set_postfix(**info)

            if agent.is_logging:
                wandb.log(info)

    agent.save(ver=0)
    return agent, scores


def test_agent(agent, envs, render=False):
    scores = []
    episode = 0
    n_episodes = envs.num_envs  # execute approximately 1 episode per env
    s_t = envs.reset()
    while episode < n_episodes:
        if render:
            envs.render()
        a_t = agent.act(s_t, training=False).actions
        s_tp1, _, _, info = envs.step(a_t)
        s_t = s_tp1
        for single_step in info:
            if "episode" in single_step.keys():
                scores.append(single_step['episode']['r'])
                episode += 1
    return np.array(scores)


def load_agent(algo, path, ver):
    if algo == 'dqn':
        agent = agents.DQNAgent.load(path, ver=ver, epsilon=0.00, training=False)
    elif algo == 'vpg':
        agent = agents.VPG.load(path, ver=ver, training=False)
    elif algo == 'a2c':
        agent = agents.A2C.load(path, ver=ver, training=False)
    elif algo == 'ddpg':
        agent = agents.DDPG.load(path, ver=ver, training=False)
    elif algo == 'ppo':
        agent = agents.PPO.load(path, ver=ver, training=False)
    elif algo == 'sac':
        agent = agents.SAC.load(path, ver=ver, training=False)
    else:
        raise ValueError(f'unsupported algorithm {algo}')
    return agent


def reset_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
