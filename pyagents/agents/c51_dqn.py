from typing import Optional, List, Dict
import gin
import numpy as np
import tensorflow as tf
import wandb
from keras.losses import MeanSquaredError, Huber

from pyagents.agents.dqn import DQNAgent
from pyagents.memory import load_memories
from pyagents.networks import C51QNetwork
from pyagents.policies import QPolicy, EpsGreedyPolicy
from copy import deepcopy


@gin.configurable
class C51DQNAgent(DQNAgent):
    """An agent implementing a DQN algorithm based on the C51 algorithm (https://arxiv.org/pdf/1707.06887.pdf).
       Only supports Discrete action spaces."""

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 q_network: C51QNetwork,
                 v_min: Optional[float] = None,
                 v_max: float = 200.0,
                 optimizer: tf.keras.optimizers.Optimizer = None,
                 gamma: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.99,
                 epsilon_min: float = 0.01,
                 target_update_period: int = 500,
                 tau: float = 1.0,
                 ddqn: bool = True,
                 buffer: Optional = 'uniform',
                 gradient_clip_norm: Optional[float] = 1.0,
                 normalize_obs: bool = True,
                 reward_scaling: float = 1.0,
                 log_dict: dict = None,
                 name: str = 'C51DQNAgent',
                 training: bool = True,
                 save_dir: str = './output',
                 wandb_params: Optional[dict] = None,
                 dtype='float32'):
        """Creates a C51 DQN agent. Number of atoms is taken from the network.

        Args:
            state_shape: Tuple representing the shape of the input space (observations).
            action_shape: Tuple representing the shape of the action space.
            q_network: network trained by the agent.
            v_min: (Optional) minimum value of the return distribution. If None, defaults to -v_max.
            v_max: maximum value of the return distribution.
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
            ddqn: (Optional) if True, uses Double DQN loss. Defaults to True.
            buffer: (Optional) buffer to store memories. Defaults to a uniform buffer.
            normalize_obs: (Optional) if True, keeps track of running statistics to normalize observations.
              Defaults to True.
            reward_scaling: (Optional) scale factor applied to the reward. Defaults to 1.0.
            loss_fn: (Optional) loss function, either 'mse' or 'huber'. Defaults to 'mse'.
            gradient_clip_norm: (Optional) If provided, gradients are scaled so that
              the norm does not exceed this value. Defaults to 0.5.
            log_dict: (Optional) additional logger parameters.
            """
        super(C51DQNAgent, self).__init__(state_shape,
                                          action_shape,
                                          q_network=q_network,
                                          optimizer=optimizer,
                                          gamma=gamma,
                                          epsilon=epsilon,
                                          epsilon_decay=epsilon_decay,
                                          epsilon_min=epsilon_min,
                                          target_update_period=target_update_period,
                                          tau=tau,
                                          ddqn=ddqn,
                                          normalize_obs=normalize_obs,
                                          reward_scaling=reward_scaling,
                                          training=training,
                                          buffer=buffer,
                                          save_dir=save_dir,
                                          name=name,
                                          dtype=dtype)
        if optimizer is None and training:
            raise ValueError('agent cannot be trained without optimizer')

        assert isinstance(q_network, C51QNetwork), 'distributional dqn agent requires distributional q net'
        self._online_q_network = q_network
        self._target_q_network = deepcopy(self._online_q_network)
        self._target_update_period = target_update_period
        self._optimizer = optimizer
        self._train_step = 0
        self._gradient_clip_norm = gradient_clip_norm
        self._name = name

        self.config.update({
            'v_min': v_min,
            'v_max': v_max,
        })
        if v_min is None:
            v_min = -v_max
        self._v_min = v_min
        self._v_max = v_max
        self._support = tf.cast(tf.linspace(v_min, v_max, self._online_q_network.n_atoms), self.dtype)
        self._delta_z = (v_max - v_min) / (self._online_q_network.n_atoms - 1)
        self._online_q_network.set_support(self._support)
        self._target_q_network.set_support(self._support)

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
        # build networks
        self._online_q_network(tf.ones((1, *state_shape)))
        self._target_q_network(tf.ones((1, *state_shape)))

    def _wandb_define_metrics(self):
        super()._wandb_define_metrics()

    def _loss(self, memories, weights=None):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memories
        # boolean mask to choose only performed actions, assumes 1d action space
        mask = tf.one_hot(action_batch, self.action_shape, on_value=True, off_value=False)

        online_out = self._online_q_network(state_batch)
        preds_qvals_logits = online_out.logits[mask]
        target_out = self._target_q_network(next_state_batch)
        next_target_dist = target_out.dist_params

        if self._ddqn:
            # double q-learning: select actions using online network, and evaluate them using target network
            self._online_q_network.reset_noise()
            next_online_out = self._online_q_network(next_state_batch)
            actions = tf.one_hot(next_online_out.actions, self.action_shape, on_value=True, off_value=False)
        else:
            # select actions from target network
            actions = tf.one_hot(target_out.actions, self.action_shape, on_value=True, off_value=False)
        next_target_dist = next_target_dist[actions]

        proj_dist, targets = self._project_dist(next_target_dist, reward_batch, done_batch)

        td_residuals = tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(proj_dist),
                                                               logits=preds_qvals_logits)
        if weights is not None:  # weigh each sample by its weight before mean reduction
            td_residuals = weights * td_residuals
        loss = tf.reduce_mean(td_residuals)

        # logging
        q_values = tf.stop_gradient(online_out.critic_values[mask])
        q_targets = tf.stop_gradient(tf.tensordot(proj_dist, self._support, 1))
        return {'loss': loss,
                'td_residuals': td_residuals,
                'q_values': q_values,
                'q_targets': q_targets}

    def _project_dist(self, p_astar, rewards, dones):
        """Computes projection from transition. See Algorithm 1 in Bellemare et al., 2017"""
        batch_size = rewards.shape[0]
        n_atoms = self._online_q_network.n_atoms

        # needed for vectorization along batch dim
        tiled_support = tf.reshape(tf.tile(self._support, (batch_size,)), (batch_size, n_atoms))
        # compute projection
        T_z = rewards + (1 - dones) * self._gamma * tiled_support
        T_z = tf.clip_by_value(T_z, self._v_min, self._v_max)

        b = (T_z - self._v_min) / self._delta_z
        l, u = tf.math.floor(b), tf.math.ceil(b)
        lb_term = p_astar * (u - b)
        ub_term = p_astar * (b - l)

        m = tf.zeros((batch_size, n_atoms), dtype=self.dtype)
        for j in tf.range(n_atoms):
            update_idx = tf.where(tf.one_hot(tf.cast(l[:, j], tf.int32), n_atoms, on_value=True, off_value=False))
            m = tf.tensor_scatter_nd_add(m, update_idx, lb_term[:, j])
            update_idx = tf.where(tf.one_hot(tf.cast(u[:, j], tf.int32), n_atoms, on_value=True, off_value=False))
            m = tf.tensor_scatter_nd_add(m, update_idx, ub_term[:, j])

        return m, T_z

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
        done_batch = np.array([sample[4] for sample in minibatch]).reshape((-1, 1))
        return [states_batch, action_batch, reward_batch, next_states_batch, done_batch]

    @staticmethod
    def networks_name(config: dict) -> List[tuple]:
        return [('q_net', C51QNetwork)]
