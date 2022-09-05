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
                 critic1_opt: tf.keras.optimizers.Optimizer = None,
                 critic2: Optional[QNetwork] = None,
                 critic2_opt: tf.keras.optimizers.Optimizer = None,
                 alpha_opt: Optional[tf.keras.optimizers.Optimizer] = None,
                 policy: Policy = None,
                 buffer: Optional[Buffer] = None,
                 gamma: float = 0.99,
                 standardize: bool = True,
                 reward_normalization: bool = True,
                 reward_scaling: float = 1.,
                 target_update_period: int = 500,
                 tau: float = 1.0,
                 initial_alpha: float = 1.,
                 train_alpha: bool = True,
                 target_entropy: Optional[float] = None,
                 gradient_clip_norm: Optional[float] = 0.5,
                 normalize_obs: bool = True,
                 training: bool = True,
                 log_dict: dict = None,
                 name: str = 'SAC',
                 save_dir: str = './output',
                 wandb_params: Optional[dict] = None,
                 dtype: str = 'float32'):
        super(SAC, self).__init__(state_shape,
                                  action_shape,
                                  training=training,
                                  normalize_obs=normalize_obs,
                                  buffer=buffer,
                                  save_dir=save_dir,
                                  name=name,
                                  dtype=dtype)
        if (actor_opt is None or critic1_opt is None or critic2_opt is None) and training:
            raise ValueError('agent cannot be trained without optimizers')

        self._actor = actor
        self._online_critic1 = critic
        self._target_critic1 = deepcopy(self._online_critic1)
        self._online_critic2 = critic2
        if self._online_critic2 is None:
            self._online_critic2 = deepcopy(self._online_critic1)
        self._target_critic2 = deepcopy(self._online_critic2)
        self._actor_opt = actor_opt
        self._critic1_opt = critic1_opt
        self._critic2_opt = critic2_opt

        self._initial_alpha = initial_alpha
        self.train_alpha = train_alpha
        self._log_alpha = tf.Variable(initial_value=tf.math.log(initial_alpha), trainable=True)
        self._alpha_opt = alpha_opt
        if self.train_alpha and self._alpha_opt is None and self._actor_opt is not None:
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
        self.target_entropy = target_entropy if target_entropy else -np.prod(self.action_shape)
        if reward_normalization:
            self.init_normalizer('reward', (1,))
        self.reward_scale = reward_scaling
        self.config.update({'gamma': self.gamma,
                            'tau': self.tau,
                            'target_update_period': self.target_update_period,
                            'standardize': self._standardize,
                            'reward_normalization': reward_normalization,
                            'gradient_clip_norm': self._gradient_clip_norm,
                            'target_entropy': self.target_entropy,
                            'initial_alpha': self._initial_alpha,
                            'train_alpha': train_alpha,
                            'reward_scaling': reward_scaling})

        if wandb_params:
            self._init_logger(wandb_params,
                              {**self.config,
                               **{f'actor/{k}': v for k, v in self._actor.get_config().items()},
                               **{f'baseline1/{k}': v for k, v in self._online_critic1.get_config().items()},
                               **{f'baseline2/{k}': v for k, v in self._online_critic2.get_config().items()},
                               **{f'buffer/{k}': v for k, v in self._memory.get_config().items()},
                               **log_dict})

        ones = tf.ones((1, *state_shape))
        a = self._actor(ones).actions
        self._online_critic1((ones, a))
        self._target_critic1((ones, a))
        self._online_critic2((ones, a))
        self._target_critic2((ones, a))

    @property
    def alpha(self):
        return tf.exp(tf.stop_gradient(self._log_alpha))

    def remember(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool, *args,
                 **kwargs) -> None:
        if 'reward' in self.normalizers:
            self.update_normalizer('reward', reward)
        super().remember(state, action, reward, next_state, done, *args, **kwargs)

    def _wandb_define_metrics(self):
        super()._wandb_define_metrics()
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic1_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic2_loss', step_metric="train_step", summary="min")
        wandb.define_metric('alpha_loss', step_metric="train_step", summary="min")

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
        if 'reward' in self.normalizers:
            _, std_r = self.get_normalizer('reward')
            reward_batch /= std_r
        reward_batch = tf.expand_dims(reward_batch, 1)  # shape must be (batch, 1)
        done_batch = tf.cast(tf.convert_to_tensor([sample[4] for sample in minibatch]), self.dtype)
        done_batch = tf.expand_dims(done_batch, 1)  # shape must be (batch, 1)
        return [states_batch, action_batch, reward_batch, next_states_batch, done_batch]

    def _loss(self, *args, **kwargs):
        pass

    def _loss_q(self, q_net, states, actions, targets) -> dict:
        q = q_net((states, actions)).critic_values
        q_td_loss = 0.5 * tf.math.squared_difference(q, targets)
        q_loss = tf.reduce_mean(q_td_loss)
        return {'critic_loss': q_loss,
                'critic_td_loss': q_td_loss,
                'q': q}

    def _loss_pi(self, states) -> dict:
        # alpha log pi - q due to gradient ascent
        act_out = self._actor(states)
        actions = act_out.actions
        logprobs = act_out.logprobs

        q = tf.reduce_mean([self._online_critic1((states, actions)).critic_values,
                            self._online_critic2((states, actions)).critic_values], axis=0)
        act_loss = tf.reduce_mean(self.alpha * tf.expand_dims(logprobs, 1) - q)
        return {'act_loss': act_loss, 'logprobs': logprobs}

    def _train(self, batch_size: int, update_rounds: int, *args, **kwargs) -> dict:
        assert self._training, 'called train function while in evaluation mode, call toggle_training() before'
        assert len(self._memory) > batch_size, f'batch size bigger than amount of memories'
        memories, indexes, is_weights = self._memory.sample(batch_size, vectorizing_fn=self._minibatch_to_tf)
        states, actions, rewards, next_states, dones = memories  # (s, a, r, s', d)
        assert rewards.shape == (batch_size, 1), f"expected rewards with shape (batch, 1), received: {rewards.shape}"
        assert dones.shape == (batch_size, 1), f"expected rewards with shape (batch, 1), received: {dones.shape}"

        # compute targets
        act_out = self._actor(next_states)
        next_action = act_out.actions
        q1_target = self._target_critic1((next_states, next_action)).critic_values
        q2_target = self._target_critic2((next_states, next_action)).critic_values
        targets = self.reward_scale * rewards + self.gamma * (1 - dones) * (
                tf.minimum(q1_target, q2_target) - self.alpha * tf.expand_dims(act_out.logprobs, 1))

        critic_loss_info = dict()
        for i, (q, q_opt) in enumerate(zip([self._online_critic1, self._online_critic2],
                                           [self._critic1_opt, self._critic2_opt])):
            critic_vars = q.trainable_variables
            with tf.GradientTape(watch_accessed_variables=False) as q_tape:
                q_tape.watch(critic_vars)
                info = self._loss_q(q, states, actions, tf.stop_gradient(targets))
                critic_loss = info['critic_loss']

            critic_grads_and_vars, critic_norm = self._apply_gradients(critic_loss, q_opt, critic_vars, q_tape)
            critic_loss_info[f'critic{i + 1}_loss'] = critic_loss
            critic_loss_info[f'critic{i + 1}_td_loss'] = info['critic_td_loss']
            critic_loss_info[f'q{i + 1}'] = info['q']
            critic_loss_info[f'critic{i + 1}_grad_vars'] = critic_grads_and_vars
            critic_loss_info[f'critic{i + 1}_norm'] = critic_norm
        # use computed loss to update memories priorities (when using a prioritized buffer) TODO verify
        critic_td_loss = 0.5 * is_weights * tf.squeeze(
            critic_loss_info['critic1_td_loss'] + critic_loss_info['critic2_td_loss'])
        self._memory.update_samples(tf.math.abs(tf.squeeze(critic_td_loss)), indexes)

        actor_vars = self._actor.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as pi_tape:
            pi_tape.watch(actor_vars)
            pi_loss_info = self._loss_pi(states)
            act_loss = pi_loss_info['act_loss']
        actor_grads_and_vars, actor_norm = self._apply_gradients(act_loss, self._actor_opt, actor_vars, pi_tape)

        if self.train_alpha:
            logprobs = pi_loss_info['logprobs']
            with tf.GradientTape(watch_accessed_variables=False) as alpha_tape:
                alpha_tape.watch([self._log_alpha])
                alpha_loss = tf.reduce_mean(
                    - tf.exp(self._log_alpha) * (tf.stop_gradient(logprobs + self.target_entropy)))
            alpha_grads_and_vars, alpha_norm = self._apply_gradients(alpha_loss, self._alpha_opt,
                                                                     [self._log_alpha], alpha_tape)
            alpha_grads_log = {f'alpha/{".".join(var.name.split("/"))}': grad.numpy()
                               for grad, var in alpha_grads_and_vars}
            alpha_grads_log['alpha/norm'] = alpha_norm
        else:
            alpha_grads_log = {}

        if self._train_step % self.target_update_period == 0:
            for o, t in zip([self._online_critic1, self._online_critic2], [self._target_critic1, self._target_critic2]):
                update_target(source_vars=o.variables,
                              target_vars=t.variables,
                              tau=self.tau)
        self._train_step += 1

        critic1 = critic_loss_info['critic1_loss']
        critic2 = critic_loss_info['critic2_loss']
        loss_dict = {'policy_loss': float(act_loss),
                     'critic_loss': float(tf.reduce_mean([critic1, critic2])),
                     'critic1_loss': float(critic1),
                     'critic2_loss': float(critic2)}
        if self.train_alpha:
            loss_dict['alpha_loss'] = float(alpha_loss)
        if self.is_logging:
            if self._log_gradients:
                pi_grads_log = {f'actor/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                for grad, var in actor_grads_and_vars}
                critic1_grads_log = {f'critic1/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                     for grad, var in critic_loss_info[f'critic1_grad_vars']}
                critic2_grads_log = {f'critic2/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                     for grad, var in critic_loss_info[f'critic2_grad_vars']}
                if self._gradient_clip_norm is not None:
                    pi_grads_log['actor/norm'] = actor_norm
                    critic1_grads_log['critic1/norm'] = critic_loss_info[f'critic1_norm']
                    critic1_grads_log['critic2/norm'] = critic_loss_info[f'critic2_norm']
                self._log(do_log_step=False, prefix='gradients', **pi_grads_log, **critic1_grads_log,
                          **critic2_grads_log, **alpha_grads_log)

            state_values = tf.minimum(critic_loss_info['q1'], critic_loss_info['q2']).numpy()
            self._log(do_log_step=False, prefix='debug', state_values=state_values, td_targets=targets.numpy(),
                      alpha=float(self.alpha))
            self._log(do_log_step=True, **loss_dict, train_step=self._train_step)
        return loss_dict

    def _apply_gradients(self, loss, opt, vars, tape):
        assert not tf.math.is_inf(loss) and not tf.math.is_nan(loss)
        grads = tape.gradient(loss, vars)
        norm = None
        if self._gradient_clip_norm is not None:
            grads, norm = tf.clip_by_global_norm(grads, self._gradient_clip_norm)
        critic_grads_and_vars = list(zip(grads, vars))
        opt.apply_gradients(critic_grads_and_vars)
        return critic_grads_and_vars, norm

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
