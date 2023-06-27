from copy import deepcopy
from typing import Optional, Dict, List
import gin
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.optimizers import Adam

from pyagents.agents.off_policy_agent import OffPolicyAgent
from pyagents.agents.agent import update_target
from pyagents.memory import Buffer, load_memories
from pyagents.networks import PolicyNetwork, QNetwork
from pyagents.networks.extended_qnetwork import ExtendedQNetwork


@gin.configurable
class SACLag(OffPolicyAgent):
    """Implementation of SAC-Lag agent."""

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor: PolicyNetwork,
                 critics: ExtendedQNetwork,
                 actor_opt: tf.keras.optimizers.Optimizer = None,
                 critic_opt: tf.keras.optimizers.Optimizer = None,
                 alpha_opt: Optional[tf.keras.optimizers.Optimizer] = None,
                 lambda_opt: Optional[tf.keras.optimizers.Optimizer] = None,
                 buffer: Optional[Buffer] = None,
                 gamma: float = 0.99,
                 standardize: bool = True,
                 reward_shape: tuple = (1,),
                 reward_normalization: bool = True,
                 reward_scaling: float = 1.,
                 initial_lambda=1.0,
                 train_lambda: bool = True,
                 target_update_period: int = 500,
                 tau: float = 1.0,
                 initial_alpha: float = 1.,
                 train_alpha: bool = True,
                 target_entropy: Optional[float] = None,
                 gradient_clip_norm: Optional[float] = 0.5,
                 normalize_obs: bool = True,
                 training: bool = True,
                 log_dict: dict = None,
                 save_memories: bool = False,
                 name: str = 'SACSE',
                 save_dir: str = './output',
                 wandb_params: Optional[dict] = None,
                 dtype: str = 'float32'):
        super().__init__(state_shape,
                         action_shape,
                         training,
                         gamma,
                         buffer,
                         normalize_obs,
                         reward_scaling,
                         save_dir,
                         save_memories,
                         name,
                         dtype)

        if (actor_opt is None or critic_opt is None) and training:
            raise ValueError('agent cannot be trained without optimizers')

        self._actor = actor

        self._online_critics = critics
        self._target_critics = deepcopy(self._online_critics)

        self._actor_opt = actor_opt
        self._critic_opt = critic_opt

        self._initial_alpha = initial_alpha
        self.train_alpha = train_alpha
        self._log_alpha = tf.Variable(initial_value=tf.math.log(initial_alpha), trainable=True)
        self._alpha_opt = alpha_opt
        if self.train_alpha and self._alpha_opt is None and self._actor_opt is not None:
            self._alpha_opt = Adam(learning_rate=self._actor_opt.learning_rate)

        self._lambda_weight = tf.Variable(initial_value=tf.math.log(tf.math.exp(initial_lambda) - 1.), trainable=True)
        self.train_lambda = train_lambda
        self._lambda_opt = lambda_opt
        if self.train_lambda and self._lambda_opt is None and self._actor_opt is not None:
            self._lambda_opt = Adam(learning_rate=self._actor_opt.learning_rate)

        self._policy = self._actor.get_policy()

        self.gamma = gamma
        self._standardize = standardize
        self.target_update_period = target_update_period
        self.tau = tau
        self._gradient_clip_norm = gradient_clip_norm
        self.target_entropy = target_entropy if target_entropy else -np.prod(self.action_shape)

        self.reward_shape = reward_shape
        if reward_normalization:
            self.init_normalizer('reward', reward_shape)
        self.reward_scale = reward_scaling

        self.config.update({'gamma': self.gamma,
                            'tau': self.tau,
                            'target_update_period': self.target_update_period,
                            'standardize': self._standardize,
                            'n_critics': self._online_critics.n_critics,
                            'reward_shape': reward_shape,
                            'initial_lambda': initial_lambda,
                            'train_lambda': train_lambda,
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
                               **{f'critics/{k}': v for k, v in self._online_critics.get_config().items()},
                               **{f'buffer/{k}': v for k, v in self._memory.get_config().items()},
                               **log_dict})

        ones = tf.ones((1, *state_shape))
        a = self._actor(ones).actions
        self._online_critics((ones, a))
        self._target_critics((ones, a))

    def init(self, envs, env_config=None, min_memories=None, actions=None, *args, **kwargs):
        self.num_envs = getattr(envs, "num_envs", 1)
        if self._wandb_run is not None and env_config is not None:
            self._wandb_run.config.update(env_config)
        if min_memories is None:
            min_memories = self._memory.get_config()['size']
        s_t, _ = envs.reset()
        for _ in range(min_memories // self.num_envs):
            a_t = envs.action_space.sample()
            s_tp1, r_t, terminated, truncated, info = envs.step(a_t)
            if 'final_info' in info:
                assert 'cost' not in info, 'issue with ending states on vector env'  # TODO check if it works
                envs_over = np.logical_or(terminated, truncated).reshape(-1, 1)
                s_tp1_with_final = np.where(envs_over, np.stack(info['final_observation']), s_tp1)
                info = info['final_info'][envs_over]
                all_costs = [info['final_info'][i].get('constraint_violation', 0.0) for i in range(self.num_envs)]
                r_t = np.stack([r_t, all_costs], axis=1)
                self.remember(state=s_t,
                              action=a_t,
                              reward=r_t,
                              next_state=s_tp1_with_final,
                              done=terminated)
            else:
                r_t = np.stack([r_t, info['constraint_violation']], axis=1)
                self.remember(state=s_t,
                              action=a_t,
                              reward=r_t,
                              next_state=s_tp1,
                              done=terminated)
            s_t = s_tp1

    @property
    def alpha_actor(self):
        return tf.exp(tf.stop_gradient(self._log_alpha))

    @property
    def lambda_weight(self):
        return tf.math.softplus(tf.stop_gradient(self._lambda_weight))

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
        wandb.define_metric('lambda_loss', step_metric="train_step", summary="min")

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
        # reward_batch = tf.expand_dims(reward_batch, 1)  # shape must be (batch, 1)
        done_batch = tf.cast(tf.convert_to_tensor([sample[4] for sample in minibatch]), self.dtype)
        done_batch = tf.expand_dims(done_batch, 1)  # shape must be (batch, 1)
        return [states_batch, action_batch, reward_batch, next_states_batch, done_batch]

    def _loss(self, *args, **kwargs):
        pass

    def _loss_q(self, q_net, states, actions, targets) -> dict:
        targets = tf.expand_dims(targets, axis=-1)
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
        # utility q values
        q = tf.reduce_mean(self._online_critics((states, actions)).critic_values, axis=1)[..., 0]  # b,
        act_loss = tf.reduce_mean(self.alpha_actor * logprobs - q)
        return {'act_loss': act_loss, 'logprobs': logprobs}

    def _train(self, batch_size: int, update_rounds: int, *args, **kwargs) -> dict:
        assert self._training, 'called train function while in evaluation mode, call toggle_training() before'
        assert len(self._memory) > batch_size, f'batch size bigger than amount of memories'
        memories, indexes, is_weights = self._memory.sample(batch_size, vectorizing_fn=self._minibatch_to_tf)
        states, actions, rewards, next_states, dones = memories  # (s, a, r, s', d)
        assert rewards.shape == (
            batch_size, 2), f"expected rewards with shape (batch, 2), received: {rewards.shape}"
        assert dones.shape == (batch_size, 1), f"expected rewards with shape (batch, 1), received: {dones.shape}"

        # compute targets
        act_out = self._actor(next_states)
        next_action = act_out.actions
        q_target = self._target_critics((next_states, next_action)).critic_values  # b, n_crit, rew_shape
        targets = self.reward_scale * rewards + self.gamma * (1 - dones) * tf.math.reduce_min(q_target,
                                                                                              axis=1)  # b, rew_shape

        critic_loss_info = dict()
        for name, q in self._online_critics.networks.items():
            reward_index = int(name[-1])
            critic_vars = q.trainable_variables
            with tf.GradientTape(watch_accessed_variables=False) as q_tape:
                q_tape.watch(critic_vars)
                info = self._loss_q(q, states, actions, tf.stop_gradient(targets[..., reward_index]))
                critic_loss = info['critic_loss']

            critic_grads_and_vars, critic_norm = self._apply_gradients(critic_loss, self._critic_opt,
                                                                       critic_vars, q_tape)
            critic_loss_info[f'critic{name[4:]}_loss'] = critic_loss
            critic_loss_info[f'critic{name[4:]}_td_errors'] = info['critic_td_loss']
            critic_loss_info[f'q{name[4:]}'] = info['q']
            critic_loss_info[f'critic{name[4:]}_grad_vars'] = critic_grads_and_vars
            critic_loss_info[f'critic{name[4:]}_norm'] = critic_norm

        actor_vars = self._actor.trainable_variables

        with tf.GradientTape(watch_accessed_variables=False) as pi_tape:
            pi_tape.watch(actor_vars)
            pi_loss_info = self._loss_pi(states)
            act_loss = pi_loss_info['act_loss']

        actor_grads_and_vars, actor_norm = self._apply_gradients(act_loss, self._actor_opt, actor_vars, pi_tape)

        # TODO finire con stats
        critic_losses = {c: float(l) for c, l in critic_loss_info.items() if c.endswith('loss')}
        loss_dict = {'policy_loss': float(act_loss),
                     **critic_losses}

        if self.train_alpha:
            logprobs = pi_loss_info['logprobs']
            with tf.GradientTape(watch_accessed_variables=False) as alpha_tape:
                alpha_tape.watch([self._log_alpha])
                alpha_loss = tf.reduce_mean(
                    - tf.exp(self._log_alpha) * (tf.stop_gradient(logprobs + self.target_entropy)))
                alpha_grads_and_vars, alpha_norm = self._apply_gradients(alpha_loss, self._alpha_opt,
                                                                         [self._log_alpha], alpha_tape)
                loss_dict[f'loss_alpha'] = alpha_loss
                alpha_grads_log = {f'alpha/{".".join(var.name.split("/"))}': grad.numpy()
                                   for grad, var in alpha_grads_and_vars}
                alpha_grads_log[f'alpha/norm'] = alpha_norm
        else:
            alpha_grads_log = {}

        if self.train_lambda:
            with tf.GradientTape(watch_accessed_variables=False) as lambda_tape:
                lambda_tape.watch([self._lambda_weight])
                lambda_loss = tf.math.softplus(self._lambda_weight) * tf.reduce_mean(rewards[..., 1])
            lambda_grads_and_vars, lambda_norm = self._apply_gradients(lambda_loss, self._lambda_opt,
                                                                       [self._lambda_weight], lambda_tape)
            loss_dict['lambda_loss'] = lambda_loss
            lambda_grads_log = {f'lambda/{".".join(var.name.split("/"))}': grad.numpy()
                                for grad, var in lambda_grads_and_vars}
            lambda_grads_log[f'lambda/norm'] = lambda_norm
        else:
            lambda_grads_log = {}

        if self._train_step % self.target_update_period == 0:
            for o, t in zip(self._online_critics.networks.values(),
                            self._target_critics.networks.values()):
                update_target(source_vars=o.variables,
                              target_vars=t.variables,
                              tau=self.tau)
        self._train_step += 1

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
                          **critic2_grads_log, **alpha_grads_log, **lambda_grads_log)

            # state_values = tf.minimum(critic_loss_info['q1'], critic_loss_info['q2']).numpy()
            self._log(do_log_step=False, prefix='debug', td_targets=targets.numpy(),
                      alpha_actor=float(self.alpha_actor),
                      lambda_weight=float(self.lambda_weight))
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
        return [('actor', self._actor.get_config(), self._actor.get_weights())] \
            + [(name, net.get_config(), net.get_weights()) for name, net in self._online_critics.networks.items()]

    @staticmethod
    def networks_name(config: dict) -> List[tuple]:
        return [('actor', PolicyNetwork)] + [(f'QNet{c}_{r}', QNetwork)
                                              for r in range(config['reward_shape'][0])
                                              for c in range(config['n_critics'])]

    @staticmethod
    def generate_input_config(
            agent_config: dict,
            networks: dict,
            load_mem: bool,
            path: str) -> Dict:
        def net_name(r, c):
            return f'QNet{c}_{r}'

        if load_mem:
            buffer = load_memories(path)
        else:
            buffer = None
        n_critics = agent_config.pop('n_critics')
        net_config = networks['QNet1_1'].get_config()
        critics_net = {net_name(r, c): networks[net_name(r, c)]
                       for r in range(agent_config['reward_shape'][0])
                       for c in range(n_critics)}
        critics = ExtendedQNetwork(**net_config, critics=critics_net, n_critics=n_critics,
                                   reward_shape=agent_config['reward_shape'])
        agent_config.update({'actor': networks['actor'],
                             'critics': critics,
                             'buffer': buffer})
        return agent_config
