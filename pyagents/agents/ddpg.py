from copy import deepcopy
from typing import Optional, List, Dict
import gin
import numpy as np
import tensorflow as tf
import wandb
from pyagents.agents.off_policy_agent import OffPolicyAgent
from pyagents.agents.agent import update_target
from pyagents.memory import Buffer, load_memories
from pyagents.networks import ACNetwork
from pyagents.policies import Policy, NormalNoisePolicy
from pyagents.utils import types


@gin.configurable
class DDPG(OffPolicyAgent):
    """ An agent implementing the Deep Deterministc Policy Gradient algorithm """

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor_critic: ACNetwork,
                 actor_opt: tf.keras.optimizers.Optimizer = None,
                 critic_opt: tf.keras.optimizers.Optimizer = None,
                 policy: Policy = None,
                 buffer: Optional[Buffer] = None,
                 action_bounds: tuple = (-np.inf, np.inf),
                 gamma: types.Float = 0.99,
                 tau: types.Float = 0.9,
                 stddev: types.Float = 0.2,
                 standardize: bool = True,
                 gradient_clip_norm: Optional[types.Float] = 0.5,
                 training: bool = True,
                 log_dict: dict = None,
                 save_dir: str = './output',
                 save_memories: bool = False,
                 name: str = 'DDPG',
                 wandb_params: Optional[dict] = None):
        """Creates a DDPG agent.

        Args:
            state_shape: Tuple representing the shape of the input space (observations).
            action_shape: Tuple representing the shape of the action space.
            actor_critic: A network comprising both a policy and a value function.
            actor_opt: Optimizer to use for training.
            critic_opt: Optimizer used to train the Q function.
            policy: (Optional) The policy the agent should follow. Defaults to policy network's policy.
            gamma: (Optional) Discount factor. Defaults to 0.5.
            standardize: (Optional) If True, standardizes delta before computing gradient update. Defaults to True.
            gradient_clip_norm: (Optional) Global norm for gradient clipping, pass None to disable. Defaults to 0.5.
            training: (Optional) If True, agent is in training phase. Defaults to True.
            log_dict: (Optional) Additional dict of config parameters that should be logged by WandB. Defaults to None.
            name: (Optional) Name of the agent.
            wandb_params: (Optional) Dict of parameters to enable WandB logging. Defaults to None.
            """
        super(DDPG, self).__init__(state_shape,
                                   action_shape,
                                   training=training,
                                   buffer=buffer,
                                   save_dir=save_dir,
                                   save_memories=save_memories,
                                   name=name)
        if actor_opt is None or critic_opt is None and training:
            raise ValueError('agent cannot be trained without optimizers')
        self._ac = actor_critic
        self._ac_target = deepcopy(self._ac)
        self._actor_opt = actor_opt
        self._critic_opt = critic_opt
        self._critic_loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

        if policy is None:
            p = self._ac.get_policy()
            self._policy = NormalNoisePolicy(p, stddev=stddev, bounds=action_bounds)
        else:
            self._policy = policy

        self._action_bounds = action_bounds
        self.gamma = gamma
        self._tau = tau
        self._standardize = standardize
        self._gradient_clip_norm = gradient_clip_norm

        self.config.update({
            'gamma': self.gamma,
            'tau': self._tau,
            'standardize': self._standardize,
            'gradient_clip_norm': self._gradient_clip_norm,
        })

        if wandb_params:
            self._init_logger(wandb_params,
                              {**self.config,
                               **{f'actor_critic/{k}': v for k, v in self._ac.get_config().items()},
                               **log_dict})
        self._ac(tf.ones((1, *state_shape))), self._ac_target(tf.ones((1, *state_shape)))  # TODO remove

    def _wandb_define_metrics(self):
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")

    def _minibatch_to_tf(self, minibatch):
        """ Given a list of experience tuples (s_t, a_t, r_t, s_t+1, done_t)
            returns list of 5 tensors batches """
        state_batch = tf.convert_to_tensor([sample[0].reshape(self.state_shape)
                                            for sample in minibatch], dtype=tf.float32)
        action_batch = tf.convert_to_tensor([sample[1] for sample in minibatch], dtype=tf.float32)
        reward_batch = tf.convert_to_tensor([sample[2] for sample in minibatch], dtype=tf.float32)
        reward_batch = tf.expand_dims(reward_batch, 1)  # shape must be (batch, 1)
        new_state_batch = tf.convert_to_tensor([sample[3].reshape(self.state_shape)
                                                for sample in minibatch], dtype=tf.float32)
        done_batch = tf.convert_to_tensor([sample[4] for sample in minibatch], dtype=tf.float32)
        done_batch = tf.expand_dims(done_batch, 1)  # shape must be (batch, 1)
        return [state_batch, action_batch, reward_batch, new_state_batch, done_batch]

    def _train(self, batch_size=None, *args, **kwargs):
        self._memory.commit_ltmemory()
        assert self._training, 'called train function but agent is in evaluation mode'
        memories, indexes, is_weights = self._memory.sample(batch_size, vectorizing_fn=self._minibatch_to_tf)
        states, actions, rewards, next_states, dones = memories  # (s, a, r, s', d)
        assert rewards.shape == (batch_size, 1), f"expected rewards with shape (batch, 1), received: {rewards.shape}"
        assert dones.shape == (batch_size, 1), f"expected rewards with shape (batch, 1), received: {dones.shape}"

        # Compute targets
        act = self._ac_target.pi(next_states) * self._ac_target.pi.get_policy().scaling_factor  # TODO remove and do it better
        q_preds = self._ac_target.critic((next_states, act))  # Q_targ(s', pi_targ(s'))
        targets = rewards + self.gamma * (1 - dones) * q_preds

        # Gradient descent step for Q function on MSBE loss
        with tf.GradientTape(watch_accessed_variables=False) as q_tape:
            q_tape.watch(self._ac.critic.trainable_variables)
            q_loss = (tf.stop_gradient(targets) - self._ac.critic((states, actions))) ** 2
            q_loss = tf.reduce_mean(q_loss)
        assert not tf.math.is_inf(q_loss) and not tf.math.is_nan(q_loss)
        critic_grads = q_tape.gradient(q_loss, self._ac.critic.trainable_variables)
        if self._gradient_clip_norm is not None:
            critic_grads, critic_norm = tf.clip_by_global_norm(critic_grads, self._gradient_clip_norm)
        critic_grads_and_vars = list(zip(critic_grads, self._ac.critic.trainable_variables))
        self._critic_opt.apply_gradients(critic_grads_and_vars)

        # Gradient ascent step for policy
        factor = self._ac.pi.get_policy().scaling_factor   # TODO remove and do it better
        with tf.GradientTape(watch_accessed_variables=False) as pi_tape:
            pi_tape.watch(self._ac.pi.trainable_variables)
            pi_preds = self._ac.pi(states) * factor
            pi_loss = - tf.reduce_mean(self._ac.critic((states, pi_preds)))
        assert not tf.math.is_inf(pi_loss) and not tf.math.is_nan(pi_loss)
        pi_grads = pi_tape.gradient(pi_loss, self._ac.pi.trainable_variables)
        if self._gradient_clip_norm is not None:
            pi_grads, pi_norm = tf.clip_by_global_norm(pi_grads, self._gradient_clip_norm)
        pi_grads_and_vars = list(zip(pi_grads, self._ac.pi.trainable_variables))
        self._actor_opt.apply_gradients(pi_grads_and_vars)

        # Update target networks
        update_target(source_vars=self._ac.variables,
                      target_vars=self._ac_target.variables,
                      tau=self._tau)

        return {'policy_loss': float(pi_loss), 'critic_loss': float(q_loss)}

    def _networks_config_and_weights(self):
        return [('ac', self._ac.get_config(), self._ac.get_weights())]

    @staticmethod
    def networks_name() -> List[tuple]:
        return [('ac', ACNetwork)]

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
        net = networks['ac']
        agent_config.update({'actor_critic': net,
                             'buffer': buffer})
        return agent_config
