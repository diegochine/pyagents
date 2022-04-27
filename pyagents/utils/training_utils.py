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


@gin.configurable
def train_agent(agent, env, training_steps=10 ** 5, batch_size=64, steps_to_train=1, update_rounds=1,
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
    print('Initializing agent')
    agent.init(env, **init_params)

    if agent.is_logging:
        wandb.define_metric('score', step_metric="episode", summary="max")

    scores = []
    losses = []
    episode = 0
    training_step = 0
    env_step = 0

    with tqdm(total=training_steps, desc='TRAINING') as pbar:
        while training_step <= training_steps:
            episode += 1
            s_t = env.reset()
            s_t = np.reshape(s_t, agent.state_shape)
            episode_step = 0
            score = 0
            done = False

            while not done and episode_step < episode_max_steps:
                a_t = agent.act(s_t.reshape(1, -1))
                if isinstance(a_t, (np.ndarray, tf.Tensor)):
                    a_t = a_t[0]  # FIXME doesn't work for complex action spaces, must be more general
                s_tp1, r_t, done, info = env.step(a_t)
                s_tp1 = np.reshape(s_tp1, agent.state_shape)
                agent.remember(s_t, a_t, r_t, s_tp1, done)
                s_t = s_tp1
                score += r_t
                episode_step += 1
                env_step += 1

                # every steps_to_train env steps, perform update_rounds agent training steps
                if env_step % steps_to_train == 0:
                    for _ in range(update_rounds):
                        loss_dict = agent.train(batch_size)
                        losses.append(loss_dict)
                        training_step += 1
                        if (training_step % save_every) == 0:
                            agent.save(ver=training_step // save_every)
                    pbar.update(update_rounds)
                    pbar.set_postfix(**loss_dict)

            scores.append(score)
            pbar.set_description(f'[Episode {episode:4d}] Score: {scores[-1]:3.0f} (MA {np.mean(scores[-100:]):3.0f})')
            if agent.is_logging:
                wandb.log({'score': scores[-1], 'episode': episode})

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
