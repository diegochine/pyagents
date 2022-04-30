import os
from shutil import rmtree
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
            # s_tp1 = np.reshape(s_tp1, agent.state_shape)
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
        if episodes > 0:
            train_info['avg_return'] /= episodes
            train_info['avg_len'] /= episodes
            train_info = {**train_info, **loss_dict}
        else:
            train_info = loss_dict
        return s_t, train_info
    return train_step


def train_off_policy_agent(agent, env, batch_size=64, steps_to_train=1, update_rounds=1,
                          rollout_steps=100):
    pass


@gin.configurable
def train_agent(agent, envs, training_steps=10 ** 5, batch_size=64, steps_to_train=1, update_rounds=1, rollout_steps=100,
                episode_max_steps=500, init_params=None, output_dir="./output/", save_every=1000):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        rmtree(output_dir)
        os.makedirs(output_dir)

    if init_params is not None:
        assert isinstance(init_params, dict), f'init_params should be dict of optional parameters for init function'
    else:
        init_params = {}

    if agent.is_logging:
        wandb.define_metric('score', step_metric="episode", summary="max")

    scores = []
    losses = []
    training_step = 0
    env_step = 0
    k = 0

    with tqdm(total=training_steps, desc='TRAINING') as pbar:
        if agent.on_policy:
            train_step_fn = train_on_policy_agent(batch_size=batch_size,
                                                  rollout_steps=rollout_steps,
                                                  update_rounds=update_rounds)
            print('Initializing agent')
            agent.init(envs, rollout_steps, **init_params)
        else:
            raise NotImplementedError()
            train_step_fn = train_off_policy_agent()
            print('Initializing agent')
            agent.init(envs, **init_params)

        state = envs.reset()
        while training_step <= training_steps:
            state, info = train_step_fn(agent, envs, s_t=state)
            training_step += update_rounds
            if training_step > k * save_every:
                agent.save(ver=training_step // save_every)
                k += 1
            pbar.update(update_rounds)
            pbar.set_postfix(**info)

            # while not done and episode_step < episode_max_steps:
            #     a_t = agent.act(s_t.reshape(1, -1))
            #     if isinstance(a_t, (np.ndarray, tf.Tensor)):
            #         a_t = a_t[0]  # FIXME doesn't work for complex action spaces, must be more general
            #     s_tp1, r_t, done, info = envs.step(a_t)
            #     s_tp1 = np.reshape(s_tp1, agent.state_shape)
            #     agent.remember(s_t, a_t, r_t, s_tp1, done)
            #     s_t = s_tp1
            #     score += r_t
            #     episode_step += 1
            #     env_step += 1
            #
            #     # every steps_to_train env steps, perform update_rounds agent training steps
            #     if env_step % steps_to_train == 0:
            #         for _ in range(update_rounds):
            #             loss_dict = agent.train(batch_size)
            #             losses.append(loss_dict)
            #             training_step += 1
            #             if (training_step % save_every) == 0:
            #                 agent.save(ver=training_step // save_every)
            #         pbar.update(update_rounds)
            #         pbar.set_postfix(**loss_dict)
            #
            # scores.append(score)

            # pbar.set_description(f'[Episode {episode:4d}] Score: {scores[-1]:3.0f} (MA {np.mean(scores[-100:]):3.0f})')
            # if agent.is_logging:
            #     wandb.log({'score': scores[-1], 'episode': episode})

    return agent, scores


def test_agent(agent, env, n_episodes=10, episode_max_steps=500, render=True):
    scores = []
    for episode in range(1, n_episodes + 1):
        s_t = env.reset()
        s_t = np.reshape(s_t, agent.state_shape)
        step = 0
        score = 0
        done = False
        while not done and step < episode_max_steps:
            if render:
                env.render()
            a_t = agent.act(s_t.reshape(1, -1))
            if isinstance(a_t, (np.ndarray, tf.Tensor)):
                a_t = a_t[0]  # FIXME doesn't work for complex action spaces, must be more general
            s_tp1, r_t, done, info = env.step(a_t)
            s_tp1 = np.reshape(s_tp1, agent.state_shape)
            s_t = s_tp1
            step += 1
            score += r_t
        scores.append(score)
        print(f'EPISODE: {episode:4d}/{n_episodes:4d}, '
              f'SCORE: {score:3.0f}')
    return scores
