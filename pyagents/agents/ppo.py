from typing import Optional, List, Dict
import gin
import numpy as np
import tensorflow as tf
import wandb
from pyagents.agents.on_policy_agent import OnPolicyAgent
from pyagents.networks import PolicyNetwork, ValueNetwork
from pyagents.policies import Policy


@gin.configurable
class PPO(OnPolicyAgent):
    """An agent implementing the PPO algorithm"""

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor: PolicyNetwork,
                 critic: ValueNetwork,
                 actor_opt: tf.keras.optimizers.Optimizer = None,
                 critic_opt: tf.keras.optimizers.Optimizer = None,
                 policy: Policy = None,
                 gamma: float = 0.99,
                 clip_eps: float = 0.2,
                 standardize: bool = True,
                 lam_gae: float = 0.9,
                 target_kl: float = 0.01,
                 returns_normalization: bool = True,
                 critic_value_coef: float = 0.5,
                 entropy_coef: float = 0,
                 gradient_clip_norm: Optional[float] = 0.5,
                 training: bool = True,
                 save_dir: str = './output',
                 log_dict: dict = None,
                 name: str = 'PPO',
                 wandb_params: Optional[dict] = None):
        """Creates an PPO agent.

                Args:
                    state_shape: Tuple representing the shape of the input space (observations).
                    action_shape: Tuple representing the shape of the action space.
                    actor: The policy network.
                    actor_opt: Optimizer to use for training the actor.
                    critic: The critic network for computing state values.
                    critic_opt: (Optional) Optimizer to use for training the critic. Must be provided when
                      a critic is also provided. Defaults to None.
                    critic_value_coef: (Optional) Coefficient applied on the value function during loss computation.
                    policy: (Optional) The policy the agent should follow. Defaults to policy network's policy.
                    gamma: (Optional) Discount factor. Defaults to 0.5.
                    standardize: (Optional) If True, standardizes advantages at the minibatch level.
                      Defaults to True.
                    lam_gae: (Optional) Lambda for Generalized Advantage Estimation. Defaults to 0.9.
                    target_kl: (Optional) Target KL divergence for early stopping condition (policy network)
                      Defaults to 0.01.
                    returns_normalization: (Optional) If True, enables return normalization. Defaults to True.
                    entropy_coef: (Optional) Coefficient applied to entropy loss. Defaults to 1e-3.
                    gradient_clip_norm: (Optional) Global norm for gradient clipping, pass None to disable.
                      Defaults to 0.5.
                    training: (Optional) If True, agent is in training phase. Defaults to True.
                    log_dict: (Optional) Additional dict of config parameters that should be logged by WandB.
                      Defaults to None.
                    name: (Optional) Name of the agent.
                    wandb_params: (Optional) Dict of parameters to enable WandB logging. Defaults to None.
                """
        super(PPO, self).__init__(state_shape, action_shape, lam_gae=lam_gae,
                                  training=training, save_dir=save_dir, name=name)
        if actor_opt is None and training:
            raise ValueError('agent cannot be trained without optimizer')

        self._pi = actor
        self._vf = critic
        self._actor_opt = actor_opt
        self._critic_opt = critic_opt

        if policy is None:
            self._policy = self._pi.get_policy()
        else:
            self._policy = policy

        self.gamma = gamma
        self._critic_value_coef = critic_value_coef
        self._entropy_coef = entropy_coef
        self._standardize = standardize
        self._gradient_clip_norm = gradient_clip_norm
        self._clip_eps = clip_eps
        self._target_kl = target_kl
        if returns_normalization:
            self.init_normalizer('returns', (1,))

        self._train_step_pi = 0
        self._train_step_v = 0

        self.config.update({
            'gamma': self.gamma,
            'clip_eps': self._clip_eps,
            'standardize': self._standardize,
            'critic_value_coef': self._critic_value_coef,
            'entropy_coef': self._entropy_coef
        })

        if wandb_params:
            if log_dict is None:
                log_dict = {}
            self._init_logger(wandb_params,
                              {**self.config,
                               **{f'actor/{k}': v for k, v in self._pi.get_config().items()},
                               **{f'baseline/{k}': v for k, v in self._vf.get_config().items()},
                               **log_dict})
        self._pi(tf.ones((1, *state_shape))), self._vf(tf.ones((1, *state_shape)))  # TODO remove

    def _wandb_define_metrics(self):
        wandb.define_metric('policy_loss', step_metric='train_step_pi', summary="min")
        wandb.define_metric('critic_loss', step_metric='train_step_v', summary="min")
        wandb.define_metric('entropy_loss', step_metric='train_step_pi', summary="min")
        wandb.define_metric('clipfrac', step_metric='train_step_pi', summary="max")
        wandb.define_metric('approx_kl', step_metric='train_step_pi', summary="max")
        wandb.define_metric('state_values', step_metric='train_step_v', summary="max")
        wandb.define_metric('td_targets', step_metric='train_step_v', summary="max")

    def _loss(self, states, actions, adv, logprobs, returns):
        policy_loss, entropy_loss, info = self._loss_pi(states=states,
                                                        actions=actions,
                                                        adv=tf.stop_gradient(adv),
                                                        logprobs=logprobs)
        critic_loss = self._loss_v(states=states,
                                   returns=tf.stop_gradient(returns))
        loss = policy_loss + critic_loss - entropy_loss
        approx_kl, clipfrac = info
        return {'loss': loss,
                'policy_loss': policy_loss,
                'entropy_loss': entropy_loss,
                'critic_loss': critic_loss,
                'approx_kl': approx_kl,
                'clipfrac': clipfrac}

    def _loss_pi(self, states, actions, adv, logprobs):
        pi_out = self._pi(inputs=states)
        dist_params = pi_out.dist_params
        new_logprobs = tf.reshape(self.get_policy().log_prob(dist_params, actions), (-1, 1))
        pi_logratio = new_logprobs - logprobs
        pi_ratio = tf.exp(pi_logratio)
        pi_clipped = tf.clip_by_value(pi_ratio, 1 - self._clip_eps, 1 + self._clip_eps)
        policy_loss = -tf.reduce_mean(tf.minimum(pi_ratio * adv, pi_clipped * adv))

        entropy_loss = self._entropy_coef * tf.reduce_mean(self.get_policy().entropy(dist_params))

        # kl divergence for early stopping condition http://joschu.net/blog/kl-approx.html and clipfrac debug variable
        approx_kl = tf.reduce_mean((pi_ratio - 1) - pi_logratio)
        clipfrac = 1 - (tf.math.count_nonzero(tf.abs((pi_ratio - 1)) < self._clip_eps) / pi_ratio.shape[0])

        return policy_loss, entropy_loss, (tf.stop_gradient(approx_kl), tf.stop_gradient(clipfrac))

    def _loss_v(self, states, returns):
        critic_values = self._vf(states).critic_values
        critic_loss = self._critic_value_coef * tf.reduce_mean((critic_values - returns) ** 2)
        return critic_loss

    def _train(self, batch_size, update_rounds, *args, **kwargs):
        # trackers for losses
        pi_losses, v_losses, e_losses = [], [], []
        # update normalizers, reshape to remove batch dimension(s)
        states = self.normalize('obs', self._memory['states'])
        states = np.reshape(states, (self._rollout_size,) + self.state_shape)
        # convert inputs to tf tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(self._memory['actions'], dtype=self.dtype)
        logprobs = tf.convert_to_tensor(self._memory['logprobs'], dtype=self.dtype)
        actions = tf.reshape(actions, (self._rollout_size, -1))
        logprobs = tf.reshape(logprobs, (self._rollout_size, -1))

        # break data in minibatches and perform updates
        indexes = np.arange(states.shape[0])
        for _ in range(update_rounds):
            # compute gae and td(lambda) returns, once per data pass
            state_values = self._vf(states).critic_values
            next_states = self.normalize('obs', self._memory['next_states'])
            next_states = np.reshape(next_states, (self._rollout_size,) + self.state_shape)
            next_states = tf.convert_to_tensor(next_states, dtype=self.dtype)
            next_state_values = self._vf(next_states).critic_values

            if 'returns' in self.normalizers:  # unnormalize value function predictions
                _, ret_std = self.get_normalizer('returns')
                state_values = state_values * ret_std
                next_state_values = next_state_values * ret_std

            unnormalized_returns, advantages = self.compute_gae(state_values=state_values, next_state_values=next_state_values)

            if 'returns' in self.normalizers:  # normalize returns and update statistics
                self.update_normalizer('returns', unnormalized_returns)
                _, ret_std = self.get_normalizer('returns')
                returns = unnormalized_returns / ret_std

            np.random.shuffle(indexes)
            # training: iterate minibatches
            for start in range(0, states.shape[0], batch_size):
                end = start + batch_size
                batch_indexes = indexes[start:end]
                mb_states = tf.gather(states, batch_indexes, axis=0)
                mb_actions = tf.gather(actions, batch_indexes, axis=0)
                mb_adv = tf.gather(advantages, batch_indexes, axis=0)
                mb_returns = tf.gather(returns, batch_indexes, axis=0)
                mb_logprobs = tf.gather(logprobs, batch_indexes, axis=0)

                if self._standardize:
                    mb_adv = ((mb_adv - tf.math.reduce_mean(mb_adv)) / (tf.math.reduce_std(mb_adv) + self.eps))

                # training: forward and loss computation
                with tf.GradientTape(persistent=True) as tape:
                    loss_info = self._loss(states=mb_states,
                                           actions=mb_actions,
                                           returns=tf.stop_gradient(mb_returns),
                                           adv=tf.stop_gradient(mb_adv),
                                           logprobs=mb_logprobs)
                loss = loss_info['loss']
                assert not tf.math.is_inf(loss) and not tf.math.is_nan(loss)

                # training: backward pass
                if loss_info['approx_kl'] <= self._target_kl:  # can still update policy network
                    actor_grads = tape.gradient(loss, self._pi.trainable_variables)
                    if self._gradient_clip_norm is not None:
                        actor_grads, actor_norm = tf.clip_by_global_norm(actor_grads, self._gradient_clip_norm)
                    actor_grads_and_vars = list(zip(actor_grads, self._pi.trainable_variables))
                    self._actor_opt.apply_gradients(actor_grads_and_vars)
                    self._train_step_pi += 1
                    if self.is_logging:
                        if self._log_gradients:
                            actor_grads_log = {f'actor/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                               for grad, var in actor_grads_and_vars}
                            if self._gradient_clip_norm is not None:
                                actor_grads_log['actor/norm'] = actor_norm
                            self._log(do_log_step=False, prefix='gradients', **actor_grads_log)
                        self._log(do_log_step=False, prefix='debug',
                                  approx_kl=loss_info['approx_kl'], clipfrac=loss_info['clipfrac'])
                        self._log(do_log_step=True, policy_loss=loss_info['policy_loss'],
                                  entropy_loss=loss_info['entropy_loss'], train_step_pi=self._train_step_pi)

                critic_grads = tape.gradient(loss, self._vf.trainable_variables)
                if self._gradient_clip_norm is not None:
                    critic_grads, critic_norm = tf.clip_by_global_norm(critic_grads, self._gradient_clip_norm)
                critic_grads_and_vars = list(zip(critic_grads, self._vf.trainable_variables))
                self._critic_opt.apply_gradients(critic_grads_and_vars)
                del tape
                self._train_step_v += 1

                # logging
                if self.is_logging:
                    if self._log_gradients:
                        critic_grads_log = {f'critic/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                            for grad, var in critic_grads_and_vars}
                        if self._gradient_clip_norm is not None:
                            critic_grads_log['critic/norm'] = critic_norm
                        self._log(do_log_step=False, prefix='gradients', **critic_grads_log)
                    self._log(do_log_step=False, prefix='debug',
                              state_values=tf.gather(state_values, batch_indexes, axis=0).numpy(),
                              td_targets=tf.gather(unnormalized_returns, batch_indexes, axis=0).numpy())
                    self._log(do_log_step=True, critic_loss=loss_info['critic_loss'], train_step_v=self._train_step_v)

                pi_losses.append(float(loss_info['policy_loss']))
                v_losses.append(float(loss_info['critic_loss']))
                e_losses.append(float(loss_info['entropy_loss']))

        self.clear_memory()

        return {'policy_loss': np.mean(pi_losses),
                'critic_loss': np.mean(v_losses),
                'entropy_loss': np.mean(e_losses)}

    def _networks_config_and_weights(self):
        a = [('actor_net', self._pi.get_config(), self._pi.get_weights())]
        if self._vf:
            a.append(('critic_net', self._vf.get_config(), self._vf.get_weights()))
        return a

    @staticmethod
    def networks_name() -> List[tuple]:
        return [('actor_net', PolicyNetwork), ('critic_net', ValueNetwork)]

    @staticmethod
    def generate_input_config(
            agent_config: dict,
            networks: dict,
            load_mem: bool,
            path: str) -> Dict:
        actor = networks['actor_net']
        if 'critic_net' in networks:
            critic = networks['critic_net']
        else:
            critic = None
        agent_config.update({'actor': actor,
                             'critic': critic})
        return agent_config
