from typing import Optional, List
import gin
import numpy as np
import tensorflow as tf

from pyagents.agents.dqn import DQNAgent
from pyagents.networks import IQNetwork
from pyagents.policies import QPolicy, EpsGreedyPolicy
from copy import deepcopy


@gin.configurable
class IQNAgent(DQNAgent):
    """An agent implementing the IQN algorithm (https://arxiv.org/pdf/1806.06923.pdf).
       Only supports Discrete action spaces."""

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 q_network: IQNetwork,
                 optimizer: tf.keras.optimizers.Optimizer = None,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.99,
                 epsilon_min: float = 0.01,
                 target_update_period: int = 500,
                 tau: float = 1.0,
                 kappa: float = 1.0,
                 n_samples: int = 32,
                 ddqn: bool = True,
                 buffer: Optional = 'uniform',
                 gradient_clip_norm: Optional[float] = 1.0,
                 normalize_obs: bool = True,
                 reward_scaling: float = 1.0,
                 log_dict: dict = None,
                 name: str = 'QRDQNAgent',
                 training: bool = True,
                 save_dir: str = './output',
                 wandb_params: Optional[dict] = None,
                 dtype='float32',
                 **kwargs):
        """Creates a IQN agent.

        Args:
            state_shape: Tuple representing the shape of the input space (observations).
            action_shape: Tuple representing the shape of the action space.
            q_network: network trained by the agent.
            optimizer: tf optimizer used for training.
            gamma: (Optional) discount factor. Defaults to 0.99.
            epsilon: (Optional) starting epsilon value for eps-greedy policy. Not used when noisy=True.
              Defaults to 0.1.
            epsilon_min: (Optional) minimum epsilon value for eps-greedy policy. Not used when noisy=True.
              Defaults to 0.01.
            epsilon_decay: (Optional) decay factor of epsilon value for eps-greedy policy. Not used when noisy=True.
              Defaults to 0.99.
            target_update_period: (Optional) number of training iterations after which target network is updated
              from online network. Defaults to 500.
            tau: (Optional) tau for polyak-averaging of target network update. Defaults to 1.0.
            kappa: (Optional) kappa of huber loss (either 0.0 or 1.0). Defaults to 1.0.
            n_samples: (Optional) number of iid samples used to estimate the return distribution.
            ddqn: (Optional) if True, uses Double DQN loss. Defaults to True.
            buffer: (Optional) buffer to store memories. Defaults to a uniform buffer.
            normalize_obs: (Optional) if True, keeps track of running statistics to normalize observations.
              Defaults to True.
            reward_scaling: (Optional) scale factor applied to the reward. Defaults to 1.0.
            gradient_clip_norm: (Optional) If provided, gradients are scaled so that
              the norm does not exceed this value. Defaults to 0.5.
            log_dict: (Optional) additional logger parameters.
            """
        super(IQNAgent, self).__init__(state_shape,
                                       action_shape,
                                       q_network=q_network,
                                       optimizer=optimizer,
                                       gamma=gamma,
                                       epsilon=epsilon,
                                       epsilon_decay=epsilon_decay,
                                       epsilon_min=epsilon_min,
                                       ddqn=ddqn,
                                       target_update_period=target_update_period,
                                       tau=tau,
                                       normalize_obs=normalize_obs,
                                       reward_scaling=reward_scaling,
                                       training=training,
                                       buffer=buffer,
                                       save_dir=save_dir,
                                       name=name,
                                       dtype=dtype,
                                       **kwargs)
        if optimizer is None and training:
            raise ValueError('agent cannot be trained without optimizer')

        assert isinstance(q_network, IQNetwork), 'distributional dqn agent requires distributional q net'
        self._online_q_network = q_network
        self._target_q_network = deepcopy(self._online_q_network)
        self._target_update_period = target_update_period
        self._optimizer = optimizer
        self._train_step = 0
        self._gradient_clip_norm = gradient_clip_norm
        self._name = name

        self._n_samples = n_samples
        self._kappa = kappa

        policy = QPolicy(self._state_shape, self._action_shape, self._online_q_network)
        if self._online_q_network.noisy_layers:
            self._policy = policy
        else:
            self._policy: EpsGreedyPolicy = EpsGreedyPolicy(policy, epsilon, epsilon_decay, epsilon_min)

        if wandb_params:
            self._init_logger(wandb_params,
                              {**self.config,
                               **{f'q_net/{k}': v for k, v in self._online_q_network.get_config().items()},
                               **log_dict})

    def _wandb_define_metrics(self):
        super()._wandb_define_metrics()

    def _loss(self, memories, weights=None):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memories
        # boolean mask to choose only performed actions, assumes 1d action space
        mask = tf.one_hot(action_batch, self.action_shape, on_value=True, off_value=False)

        online_out = self._online_q_network(state_batch, num_samples=self._n_samples)
        taus = online_out.dist_params
        target_out = self._target_q_network(next_state_batch, num_samples=self._n_samples)

        if self._ddqn:
            # double q-learning: select actions using online network, and evaluate them using target network
            self._online_q_network.reset_noise()
            next_online_out = self._online_q_network(next_state_batch, num_samples=self._n_samples)
            actions = tf.one_hot(next_online_out.actions, self.action_shape, on_value=True, off_value=False)
        else:
            # select actions from target network
            actions = tf.one_hot(target_out.actions, self.action_shape, on_value=True, off_value=False)
        next_zvals = tf.expand_dims(target_out.critic_values[actions], 1)

        target_zvals = reward_batch + (1. - done_batch) * self._gamma_n * tf.stop_gradient(next_zvals)
        # compute td errors for each pair of (theta_i(x), theta_j(x')) quantiles
        current_zvals = tf.tile(tf.reshape(online_out.critic_values[mask], (-1, 1, 1)), [1, 1, self._n_samples])
        target_zvals = tf.tile(tf.reshape(target_zvals, (-1, 1, 1)), [1, self._n_samples, 1])
        errors = target_zvals - current_zvals  # shape (batch, n, n)
        # compute huber loss and quantile huber loss
        abs_errors = tf.abs(errors)
        huber_loss = tf.where(abs_errors < self._kappa,
                              0.5 * tf.math.square(errors),
                              self._kappa * (abs_errors - 0.5 * self._kappa))
        quantile_huber_loss = tf.where(tf.less_equal(tf.stop_gradient(errors), 0.0),
                                       tf.abs(1 - taus) * huber_loss,  # underestimation weighted by abs(1 - tau_i)
                                       taus * huber_loss)  # overestimation weighted by tau_i

        td_residuals = tf.reduce_mean(tf.reduce_sum(quantile_huber_loss, axis=2), axis=1)

        if weights is not None:  # weigh each sample by its weight before mean reduction
            td_residuals = weights * td_residuals
        loss = tf.reduce_mean(td_residuals)

        # logging
        q_values = tf.stop_gradient(online_out.critic_values[mask])
        q_targets = tf.stop_gradient(tf.reduce_mean(target_zvals, axis=1)[:, 0])
        return {'loss': loss,
                'td_residuals': td_residuals,
                'q_values': q_values,
                'q_targets': q_targets}

    def _minibatch_to_tf(self, minibatch):
        """ Given a list of experience tuples (s_t, a_t, r_t, s_t+1, done_t)
            returns list of 5 tensors batches """
        states = np.array([sample[0].reshape(self.state_shape) for sample in minibatch])
        next_states = np.array([sample[3].reshape(self.state_shape) for sample in minibatch])
        if 'obs' in self.normalizers:
            states = self.normalize('obs', states)
            next_states = self.normalize('obs', next_states)
        states_batch = tf.convert_to_tensor(states, dtype=self.dtype)
        next_states_batch = tf.convert_to_tensor(next_states, dtype=self.dtype)
        action_batch = tf.convert_to_tensor([sample[1] for sample in minibatch], dtype=tf.int32)
        reward_batch = tf.convert_to_tensor([sample[2] for sample in minibatch], dtype=self.dtype)
        reward_batch = tf.reshape(reward_batch, (-1, 1))
        done_batch = tf.convert_to_tensor([1. if sample[4] else 0. for sample in minibatch], dtype=self.dtype)
        done_batch = tf.reshape(done_batch, (-1, 1))
        return [states_batch, action_batch, reward_batch, next_states_batch, done_batch]

    @staticmethod
    def networks_name(config: dict) -> List[tuple]:
        return [('q_net', IQNetwork)]
