import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
import gin
import wandb
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


@gin.configurable
def get_optimizer(learning_rate=0.001):
    return Adam(learning_rate=learning_rate)


def train_on_policy_agent(batch_size=128, rollout_steps=100, update_rounds=1):
    def train_step(agent, envs, s_t):
        train_info = {'avg_return': 0, 'avg_len': 0}
        episodes = 0
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
                if "episode" in single_step.keys():
                    train_info['avg_return'] += single_step['episode']['r']
                    train_info['avg_len'] += single_step['episode']['l']
                    episodes += 1

        loss_dict = agent.train(batch_size, update_rounds=update_rounds)
        train_info['train_step'] = agent.train_step
        if episodes > 0:
            train_info['avg_return'] /= episodes
            train_info['avg_len'] /= episodes
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
                if "episode" in single_step.keys():
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


def test_agent(agent, envs, render=True):
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
