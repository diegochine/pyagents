from copy import deepcopy
from typing import Optional, Dict, List
import gin
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.optimizers import Adam

from pyagents.agents.agent import update_target
from pyagents.agents.off_policy_agent import OffPolicyAgent
from pyagents.memory import Buffer, load_memories
from pyagents.networks import PolicyNetwork, QNetwork
from pyagents.policies import Policy


@gin.configurable
class SAC(OffPolicyAgent):
    """ An agent implementing the SAC algorithm handling continuous action space"""

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor: PolicyNetwork,
                 critic: QNetwork,
                 actor_opt: tf.keras.optimizers.Optimizer = None,
                 critic_opt: tf.keras.optimizers.Optimizer = None,
                 critic2: Optional[QNetwork] = None,
                 alpha_opt: Optional[tf.keras.optimizers.Optimizer] = None,
                 policy: Policy = None,
                 buffer: Optional[Buffer] = None,
                 gamma: float = 0.99,
                 standardize: bool = True,
                 reward_normalization: bool = True,
                 target_update_period: int = 500,
                 tau: float = 1.0,
                 initial_alpha: float = 1.0,
                 target_entropy: Optional[float] = None,
                 gradient_clip_norm: Optional[float] = 0.5,
                 training: bool = True,
                 log_dict: dict = None,
                 name: str = 'SAC',
                 save_dir: str = './output',
                 wandb_params: Optional[dict] = None,
                 dtype: str = 'float32'):
        super(SAC, self).__init__(state_shape,
                                  action_shape,
                                  training=training,
                                  buffer=buffer,
                                  save_dir=save_dir,
                                  name=name,
                                  dtype=dtype)
        if (actor_opt is None or critic_opt is None) and training:
            raise ValueError('agent cannot be trained without optimizers')

        self._actor = actor
        self._online_critic1 = critic
        self._target_critic1 = deepcopy(self._online_critic1)
        self._online_critic2 = critic2
        if self._online_critic2 is None:
            self._online_critic2 = deepcopy(self._online_critic1)
        self._target_critic2 = deepcopy(self._online_critic2)
        self._actor_opt = actor_opt
        self._critic_opt = critic_opt

        self._initial_alpha = initial_alpha
        self._log_alpha = tf.Variable(initial_value=tf.math.log(initial_alpha), trainable=True)
        self._alpha_opt = alpha_opt
        if self._alpha_opt is None and self._actor_opt is not None:
            self._alpha_opt = Adam(learning_rate=self._actor_opt.learning_rate)

        if policy is None:
            self._policy = self._actor.get_policy()
        else:
            self._policy = policy
        self.gamma = gamma
        self._standardize = standardize
        self.target_update_period = target_update_period
        self.tau = tau
        self._gradient_clip_norm = gradient_clip_norm
        self.target_entropy = target_entropy if target_entropy else -np.prod(self.action_shape) / 2.0
        if reward_normalization:
            self.init_normalizer('reward', (1,))

        self.config.update({'gamma': self.gamma,
                            'tau': self.tau,
                            'target_update_period': self.target_update_period,
                            'standardize': self._standardize,
                            'reward_normalization': reward_normalization,
                            'gradient_clip_norm': self._gradient_clip_norm,
                            'target_entropy': self.target_entropy,
                            'initial_alpha': self._initial_alpha})

        if wandb_params:
            self._init_logger(wandb_params,
                              {**self.config,
                               **{f'actor/{k}': v for k, v in self._actor.get_config().items()},
                               **{f'baseline1/{k}': v for k, v in self._online_critic1.get_config().items()},
                               **{f'baseline2/{k}': v for k, v in self._online_critic2.get_config().items()},
                               **{f'buffer/{k}': v for k, v in self._memory.get_config().items()},
                               **log_dict})

        ones = tf.ones((1, *state_shape))
        a = self._actor(ones).action
        self._online_critic1((ones, a))
        self._target_critic1((ones, a))
        self._online_critic2((ones, a))
        self._target_critic2((ones, a))

    @property
    def alpha(self):
        return tf.exp(tf.stop_gradient(self._log_alpha))

    def remember(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool, *args, **kwargs) -> None:
        reward = self.update_normalizer('reward', reward)
        super().remember(state, action, reward, next_state, done, *args, **kwargs)

    def _wandb_define_metrics(self):
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic1_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic2_loss', step_metric="train_step", summary="min")
        wandb.define_metric('entropy_loss', step_metric="train_step", summary="min")

    def _minibatch_to_tf(self, minibatch):
        states = np.array([sample[0].reshape(self.state_shape) for sample in minibatch])
        next_states = np.array([sample[3].reshape(self.state_shape) for sample in minibatch])
        if 'obs' in self.normalizers:
            states = self.normalize('obs', states)
            next_states = self.normalize('obs', next_states)
        states_batch = tf.convert_to_tensor(states, dtype=self.dtype)
        next_states_batch = tf.convert_to_tensor(next_states, dtype=self.dtype)
        action_batch = tf.convert_to_tensor([sample[1] for sample in minibatch], dtype=self.dtype)
        reward_batch = tf.convert_to_tensor([sample[2] for sample in minibatch], dtype=self.dtype)
        reward_batch = tf.expand_dims(reward_batch, 1)  # shape must be (batch, 1)
        done_batch = tf.cast(tf.convert_to_tensor([sample[4] for sample in minibatch]), self.dtype)
        done_batch = tf.expand_dims(done_batch, 1)  # shape must be (batch, 1)
        return [states_batch, action_batch, reward_batch, next_states_batch, done_batch]

    def _loss(self, *args, **kwargs):
        pass

    def _loss_q(self, states, actions, targets) -> dict:
        q1 = self._online_critic1((states, actions)).critic_values
        q2 = self._online_critic2((states, actions)).critic_values
        q1_td_loss = (q1 - targets) ** 2
        q1_loss = tf.reduce_mean(q1_td_loss)
        q2_td_loss = (q2 - targets) ** 2
        q2_loss = tf.reduce_mean(q2_td_loss)
        return {'critic_loss': (q1_loss + q2_loss) / 2.,
                'critic1_loss': q1_loss, 'critic2_loss': q2_loss,
                'q1': q1, 'q2': q2, 'td_loss': tf.minimum(q1_td_loss, q2_td_loss)}

    def _loss_pi(self, states) -> dict:
        # alpha log pi - q due to gradient ascent
        act_out = self._actor(states)
        actions = act_out.action
        logprobs = act_out.logprobs

        q = tf.minimum(tf.stop_gradient(self._online_critic1((states, actions)).critic_values),
                       tf.stop_gradient(self._online_critic2((states, actions)).critic_values))
        act_loss = tf.reduce_mean(self.alpha * tf.expand_dims(logprobs, 1) - q)
        return {'act_loss': act_loss, 'logprobs': logprobs}

    def _train(self, batch_size: int, update_rounds: int, *args, **kwargs) -> dict:
        self._memory.commit_ltmemory()
        assert self._training, 'called train function while in evaluation mode, call toggle_training() before'
        assert len(self._memory) > batch_size, f'batch size bigger than amount of memories'
        memories, indexes, is_weights = self._memory.sample(batch_size, vectorizing_fn=self._minibatch_to_tf)
        states, actions, rewards, next_states, dones = memories  # (s, a, r, s', d)
        assert rewards.shape == (batch_size, 1), f"expected rewards with shape (batch, 1), received: {rewards.shape}"
        assert dones.shape == (batch_size, 1), f"expected rewards with shape (batch, 1), received: {dones.shape}"

        # compute targets
        act_out = self._actor(next_states)
        next_action = act_out.action
        q1_target = self._target_critic1((next_states, next_action)).critic_values
        q2_target = self._target_critic2((next_states, next_action)).critic_values
        targets = rewards + self.gamma * (1 - dones) * (
                tf.minimum(q1_target, q2_target) - self.alpha * tf.expand_dims(act_out.logprobs, 1))

        with tf.GradientTape() as q_tape:
            critic_loss_info = self._loss_q(states, actions, tf.stop_gradient(targets))
            critic_loss = critic_loss_info['critic_loss']
        assert not tf.math.is_inf(critic_loss) and not tf.math.is_nan(critic_loss)
        critic_vars = self._online_critic1.trainable_variables + self._online_critic2.trainable_variables
        critic_grads = q_tape.gradient(critic_loss, critic_vars)
        if self._gradient_clip_norm is not None:
            critic_grads, critic_norm = tf.clip_by_global_norm(critic_grads, self._gradient_clip_norm)
        critic_grads_and_vars = list(zip(critic_grads, critic_vars))
        self._critic_opt.apply_gradients(critic_grads_and_vars)

        with tf.GradientTape() as pi_tape:
            pi_loss_info = self._loss_pi(states)
            act_loss = pi_loss_info['act_loss']
        assert not tf.math.is_inf(act_loss) and not tf.math.is_nan(act_loss)
        actor_grads = pi_tape.gradient(act_loss, self._actor.trainable_variables)
        if self._gradient_clip_norm is not None:
            actor_grads, actor_norm = tf.clip_by_global_norm(actor_grads, self._gradient_clip_norm)
        actor_grads_and_vars = list(zip(actor_grads, self._actor.trainable_variables))
        self._actor_opt.apply_gradients(actor_grads_and_vars)

        logprobs = pi_loss_info['logprobs']
        with tf.GradientTape() as alpha_tape:
            alpha_loss = - tf.reduce_mean(self._log_alpha * (tf.stop_gradient(logprobs) + self.target_entropy))
        assert not tf.math.is_inf(alpha_loss) and not tf.math.is_nan(alpha_loss)
        alpha_grads = alpha_tape.gradient(alpha_loss, [self._log_alpha])
        alpha_grads_and_vars = list(zip(alpha_grads, [self._log_alpha]))
        self._alpha_opt.apply_gradients(alpha_grads_and_vars)

        # use computed loss to update memories priorities (when using a prioritized buffer) TODO verify
        self._memory.update_samples(tf.math.abs(tf.squeeze(critic_loss_info['td_loss'])), indexes)
        if self._train_step % self.target_update_period == 0:
            for o, t in zip([self._online_critic1, self._online_critic2], [self._target_critic1, self._target_critic2]):
                update_target(source_vars=o.variables,
                              target_vars=t.variables,
                              tau=self.tau)
        self._train_step += 1

        loss_dict = {'policy_loss': float(act_loss), 'critic_loss': float(critic_loss),
                     'critic1_loss': float(critic_loss_info['critic1_loss']),
                     'critic2_loss': float(critic_loss_info['critic2_loss']),
                     'alpha_loss': float(alpha_loss)}
        if self.is_logging:
            if self._log_gradients:
                pi_grads_log = {f'actor/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                for grad, var in actor_grads_and_vars}
                critic1_grads_log = {f'critic/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                     for grad, var in critic_grads_and_vars}
                if self._gradient_clip_norm is not None:
                    pi_grads_log['actor/norm'] = actor_norm
                    critic1_grads_log['critic/norm'] = critic_norm
                self._log(do_log_step=False, prefix='gradients', **pi_grads_log, **critic1_grads_log)
            state_values = tf.minimum(critic_loss_info['q1'], critic_loss_info['q2']).numpy()
            self._log(do_log_step=False, prefix='debug', state_values=state_values, td_targets=targets.numpy())
            self._log(do_log_step=True, **loss_dict, train_step=self._train_step)
        return loss_dict

    def _networks_config_and_weights(self):
        return [('actor', self._actor.get_config(), self._actor.get_weights()),
                ('critic1', self._online_critic1.get_config(), self._online_critic1.get_weights()),
                ('critic2', self._online_critic2.get_config(), self._online_critic2.get_weights())]

    @staticmethod
    def networks_name() -> List[tuple]:
        return [('actor', PolicyNetwork),
                ('critic1', QNetwork),
                ('critic2', QNetwork)]

    @staticmethod
    def generate_input_config(
            agent_config: dict,
            networks: dict,
            load_mem: bool,
            path: str) -> Dict:
        if load_mem:
            buffer = load_memories(path)
        else:
            buffer = None
        agent_config.update({'actor': networks['actor'],
                             'critic': networks['critic1'],
                             'critic2': networks['critic2'],
                             'buffer': buffer})
        return agent_config
