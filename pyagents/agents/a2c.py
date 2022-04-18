import os
import pickle
from typing import Optional, List, Dict
import gin
import numpy as np
import tensorflow as tf
import wandb
from pyagents.agents import Agent
from pyagents.memory import load_memories
from pyagents.networks import SharedBackboneACNetwork
from pyagents.policies import Policy
from pyagents.utils import types


@gin.configurable
class A2C(Agent):
    """ An agent implementing a synchronous, deterministic version of the A3C algorithm """

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor_critic: SharedBackboneACNetwork,
                 opt: tf.keras.optimizers.Optimizer = None,
                 policy: Policy = None,
                 gamma: types.Float = 0.99,
                 standardize: bool = True,
                 n_step_return: int = 1,
                 lam_gae: float = 0.9,
                 acc_grads_steps: int = 0,
                 entropy_coef: types.Float = 1e-3,
                 gradient_clip_norm: Optional[types.Float] = 0.5,
                 training: bool = True,
                 log_dict: dict = None,
                 name: str = 'A2C',
                 save_dir: str = './output',
                 save_memories: bool = False,
                 wandb_params: Optional[dict] = None):
        """Creates an A2C agent.

        Args:
            state_shape: Tuple representing the shape of the input space (observations).
            action_shape: Tuple representing the shape of the action space.
            actor_critic: A network comprising both a policy and a value function.
            opt: Optimizer to use for training.
            policy: (Optional) The policy the agent should follow. Defaults to policy network's policy.
            gamma: (Optional) Discount factor. Defaults to 0.5.
            standardize: (Optional) If True, standardizes delta before computing gradient update. Defaults to True.
            n_step_return: (Optional) Number of steps before computing multi-step returns. Defaults to 1.
            lam_gae: (Optional) Lambda for Generalized Advantage Estimation. Defaults to 0.9.
            acc_grads_steps: (Optional) Simulates A3C asynchronous parallel learning
              by accumulating gradients over multiple backward passes. Defaults to 0 (no accumulation).
            entropy_coef: (Optional) Coefficient applied to entropy loss. Defaults to 1e-3.
            gradient_clip_norm: (Optional) Global norm for gradient clipping, pass None to disable. Defaults to 0.5.
            training: (Optional) If True, agent is in training phase. Defaults to True.
            log_dict: (Optional) Additional dict of config parameters that should be logged by WandB. Defaults to None.
            name: (Optional) Name of the agent.
            wandb_params: (Optional) Dict of parameters to enable WandB logging. Defaults to None.
        """
        super(A2C, self).__init__(state_shape,
                                  action_shape,
                                  training=training,
                                  save_dir=save_dir,
                                  save_memories=save_memories,
                                  name=name)
        if opt is None and training:
            raise ValueError('agent cannot be trained without optimizer')
        self._actor_critic = actor_critic
        self._opt = opt
        self._critic_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

        if policy is None:
            self._policy = self._actor_critic.get_policy()
        else:
            self._policy = policy

        self._n_step_return = n_step_return

        self.gamma = gamma
        self._entropy_coef = entropy_coef
        self._lam_gae = lam_gae
        self._standardize = standardize
        self._gradient_clip_norm = gradient_clip_norm

        self._trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        self._steps_last_update = 0
        self._acc_grads_steps = acc_grads_steps
        self._acc_grads = None

        self.config.update({
            'gamma': self.gamma,
            'n_step_return': self._n_step_return,
            'entropy_coef': self._entropy_coef,
            **{f'actor_critic/{k}': v for k, v in self._actor_critic.get_config().items()}
        })
        if log_dict is not None:
            self.config.update(**log_dict)

        if wandb_params:
            self._init_logger(wandb_params)

    def _wandb_define_metrics(self):
        """Defines WandB metrics.

        For A2C, they are: policy network loss; critic network loss; policy distribution entropy loss.
        """
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")
        wandb.define_metric('entropy_loss', step_metric="train_step", summary="min")

    def clear_memory(self):
        self._trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

    def remember(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool) -> None:
        self._trajectory['states'].append(state)
        self._trajectory['actions'].append(action)
        self._trajectory['rewards'].append(reward)
        self._trajectory['next_states'].append(next_state)
        self._trajectory['dones'].append(done)

    def _loss(self, memories):
        states, actions, delta = memories
        ac_out = self._actor_critic(inputs=states)
        log_prob = self._policy.log_prob(ac_out.dist_params, actions)
        entropy_loss = tf.reduce_sum(self._policy.entropy(ac_out.dist_params))
        policy_loss = -tf.reduce_sum((log_prob * (delta - tf.stop_gradient(ac_out.critic_values))))
        critic_loss = self._critic_loss_fn(delta, ac_out.critic_values)
        return policy_loss, critic_loss, entropy_loss

    def _compute_gae(self) -> tf.Tensor:
        rewards = self._trajectory['rewards']
        dones = 1 - tf.convert_to_tensor(self._trajectory['dones'], dtype=tf.float32)
        # convert states to tensors and add batch dimension
        state_values = self._actor_critic(tf.stack(self._trajectory['states'])).critic_values
        next_state_values = self._actor_critic(tf.stack(self._trajectory['next_states'])).critic_values
        returns = np.zeros(len(rewards) + 1)
        for t in reversed(range(len(rewards))):
            v_t = state_values[t]
            v_tp1 = next_state_values[t]
            delta = rewards[t] + (self.gamma * dones[t] + v_tp1) - v_t
            returns[t] = delta + (self.gamma * self._lam_gae * returns[t + 1] * dones[t])
        return tf.convert_to_tensor(returns[:-1], dtype=tf.float32)

    def _train(self, batch_size=None, *args, **kwargs) -> dict:
        if len(self._trajectory['rewards']) >= batch_size:  # perform training step
            states = tf.convert_to_tensor(self._trajectory['states'], dtype=tf.float32)
            actions = tf.convert_to_tensor(self._trajectory['actions'], dtype=tf.float32)

            delta = tf.stop_gradient(self._compute_gae())
            if self._standardize:
                delta = ((delta - tf.math.reduce_mean(delta)) / (tf.math.reduce_std(delta) + self.eps))

            # TODO shuffle minibatch?

            # forward pass and loss computation
            with tf.GradientTape() as tape:
                policy_loss, critic_loss, entropy_loss = self._loss((states, actions, tf.stop_gradient(delta)))
                loss = policy_loss + critic_loss - self._entropy_coef * entropy_loss
            assert not tf.math.is_inf(loss) and not tf.math.is_nan(loss), f'{policy_loss, critic_loss, entropy_loss}'

            # backward pass
            grads = tape.gradient(loss, self._actor_critic.trainable_variables)
            if self._gradient_clip_norm is not None:
                grads, norm = tf.clip_by_global_norm(grads, self._gradient_clip_norm)
            if self._acc_grads is None:
                self._acc_grads = np.array(grads)
            else:
                self._acc_grads += np.array(grads)

            # simulate A3C async update of gradients by multiple parallel learners
            # FIXME maybe we can remove this, feels useless, must test
            if self._steps_last_update == self._acc_grads_steps:
                self._opt.apply_gradients(list(zip(self._acc_grads, self._actor_critic.trainable_variables)))
                self._steps_last_update = 0
                self._acc_grads = None
            else:
                self._steps_last_update += 1

            self.clear_memory()
            return {'policy_loss': float(policy_loss),
                    'critic_loss': float(critic_loss),
                    'entropy_loss': float(entropy_loss)}

    def _networks_config_and_weights(self):
        net_config = self._actor_critic.get_config()
        net_weights = self._actor_critic.get_weights()
        return [('actor_critic', net_config, net_weights)]

    @staticmethod
    def networks_name() -> List[tuple]:
        return [('actor_critic', SharedBackboneACNetwork)]

    @staticmethod
    def generate_input_config(
            agent_config: dict,
            networks: dict,
            load_mem: bool,
            path: str) -> Dict:

        net = networks['actor_critic']
        agent_config.update({'actor_critic': net})
        return agent_config
