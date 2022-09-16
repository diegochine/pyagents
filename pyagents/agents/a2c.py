import os
import pickle
from typing import Optional, List, Dict
import gin
import numpy as np
import tensorflow as tf
import wandb
from pyagents.agents.on_policy_agent import OnPolicyAgent
from pyagents.networks import SharedBackboneACNetwork
from pyagents.policies import Policy


@gin.configurable
class A2C(OnPolicyAgent):
    """ An agent implementing a synchronous, deterministic version of the A3C algorithm """

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor_critic: SharedBackboneACNetwork,
                 opt: tf.keras.optimizers.Optimizer = None,
                 policy: Policy = None,
                 gamma: float = 0.99,
                 standardize: bool = True,
                 n_step_return: int = 1,
                 lam_gae: float = 0.9,
                 entropy_coef: float = 1e-3,
                 gradient_clip_norm: Optional[float] = 0.5,
                 training: bool = True,
                 log_dict: dict = None,
                 name: str = 'A2C',
                 save_dir: str = './output',
                 save_memories: bool = False,
                 wandb_params: Optional[dict] = None,
                 dtype: str = 'float32'):
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
                                  lam_gae=lam_gae,
                                  training=training,
                                  save_dir=save_dir,
                                  save_memories=save_memories,
                                  name=name,
                                  dtype=dtype)
        if opt is None and training:
            raise ValueError('agent cannot be trained without optimizer')
        self._actor_critic = actor_critic
        self._opt = opt
        self._critic_loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

        if policy is None:
            self._policy = self._actor_critic.get_policy()
        else:
            self._policy = policy

        self._n_step_return = n_step_return

        self.gamma = gamma
        self._entropy_coef = entropy_coef
        self._standardize = standardize
        self._gradient_clip_norm = gradient_clip_norm

        self.config.update({
            'gamma': self.gamma,
            'n_step_return': self._n_step_return,
            'entropy_coef': self._entropy_coef
        })

        if wandb_params:
            if log_dict is None:
                log_dict = {}
            self._init_logger(wandb_params,
                              {**self.config,
                               **{f'actor_critic/{k}': v for k, v in self._actor_critic.get_config().items()},
                               **log_dict})
        self._actor_critic(tf.ones((1, *self.state_shape)))

    def _wandb_define_metrics(self):
        """Defines WandB metrics.

        For A2C, they are: policy network loss; critic network loss; policy distribution entropy loss.
        """
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")
        wandb.define_metric('entropy_loss', step_metric="train_step", summary="min")

    def _loss(self, memories):
        states, actions, delta = memories
        ac_out = self._actor_critic(inputs=states)
        log_prob = self._policy.log_prob(ac_out.dist_params, actions)
        entropy_loss = self._entropy_coef * tf.reduce_sum(self._policy.entropy(ac_out.dist_params))
        policy_loss = -tf.reduce_sum((log_prob * (delta - tf.stop_gradient(ac_out.critic_values))))
        critic_loss = self._critic_loss_fn(delta, ac_out.critic_values)
        return policy_loss, critic_loss, entropy_loss

    def _train(self, batch_size, update_rounds, *args, **kwargs) -> dict:
        states = tf.convert_to_tensor(self._memory['states'], dtype=tf.float32)
        actions = tf.convert_to_tensor(self._memory['actions'], dtype=tf.float32)
        next_states = tf.convert_to_tensor(self._memory['next_states'], dtype=tf.float32)

        states = tf.reshape(states, (-1, *self.state_shape))
        actions = tf.reshape(actions, (-1,))
        next_states = tf.reshape(next_states, (-1, *self.state_shape))
        # GAE computation
        state_values = self._actor_critic(states).critic_values
        next_state_values = self._actor_critic(next_states).critic_values
        delta = tf.stop_gradient(self.compute_gae(state_values, next_state_values))
        if self._standardize:
            delta = ((delta - tf.math.reduce_mean(delta)) / (tf.math.reduce_std(delta) + self.eps))

        indexes = np.arange(states.shape[0])
        for _ in range(update_rounds):
            np.random.shuffle(indexes)
            for start in range(0, states.shape[0], batch_size):
                end = start + batch_size
                batch_indexes = indexes[start:end]

                # forward pass and loss computation
                s_b = tf.gather(states, batch_indexes, axis=0)
                a_b = tf.gather(actions, batch_indexes, axis=0)
                d_b = tf.gather(delta, batch_indexes, axis=0)
                with tf.GradientTape() as tape:
                    policy_loss, critic_loss, entropy_loss = self._loss((s_b, a_b, tf.stop_gradient(d_b)))
                    loss = policy_loss + critic_loss - entropy_loss
                assert not tf.math.is_inf(loss) and not tf.math.is_nan(loss), f'{policy_loss, critic_loss, entropy_loss}'

                # backward pass
                grads = tape.gradient(loss, self._actor_critic.trainable_variables)
                if self._gradient_clip_norm is not None:
                    grads, norm = tf.clip_by_global_norm(grads, self._gradient_clip_norm)
                self._opt.apply_gradients(list(zip(grads, self._actor_critic.trainable_variables)))

        self.clear_memory()
        return {'policy_loss': float(policy_loss),
                'critic_loss': float(critic_loss),
                'entropy_loss': float(entropy_loss)}

    def _networks_config_and_weights(self):
        net_config = self._actor_critic.get_config()
        net_weights = self._actor_critic.get_weights()
        return [('actor_critic', net_config, net_weights)]

    @staticmethod
    def networks_name(config: dict) -> List[tuple]:
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
