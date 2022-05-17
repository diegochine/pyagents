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
                 gamma: float = 0.99,
                 tau: float = 0.9,
                 stddev: float = 0.2,
                 standardize: bool = True,
                 gradient_clip_norm: Optional[float] = 0.5,
                 training: bool = True,
                 log_dict: dict = None,
                 save_dir: str = './output',
                 save_memories: bool = False,
                 name: str = 'DDPG',
                 wandb_params: Optional[dict] = None,
                 dtype: str = 'float32'):
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
                                   name=name,
                                   dtype=dtype)
        if (actor_opt is None or critic_opt is None) and training:
            raise ValueError('agent cannot be trained without optimizers')
        self._ac = actor_critic
        self._ac_target = deepcopy(self._ac)
        self._actor_opt = actor_opt
        self._critic_opt = critic_opt

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
            if log_dict is None:
                log_dict = {}
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
        states = np.array([sample[0].reshape(self.state_shape) for sample in minibatch])
        next_states = np.array([sample[3].reshape(self.state_shape) for sample in minibatch])
        if 'obs' in self.normalizers:
            states = self.normalize('obs', states)
            next_states = self.normalize('obs', next_states)
        states_batch = tf.convert_to_tensor(states, dtype=self.dtype)
        next_states_batch = tf.convert_to_tensor(next_states, dtype=self.dtype)
        action_batch = tf.convert_to_tensor([sample[1] for sample in minibatch], dtype=self.dtype)
        reward_batch = tf.convert_to_tensor([sample[2] for sample in minibatch], dtype=tf.float32)
        reward_batch = tf.expand_dims(reward_batch, 1)  # shape must be (batch, 1)
        done_batch = tf.cast(tf.convert_to_tensor([sample[4] for sample in minibatch]), self.dtype)
        done_batch = tf.expand_dims(done_batch, 1)  # shape must be (batch, 1)
        return [states_batch, action_batch, reward_batch, next_states_batch, done_batch]

    def _train(self, batch_size=128, *args, **kwargs):
        self._memory.commit_ltmemory()
        assert self._training, 'called train function but agent is in evaluation mode'
        memories, indexes, is_weights = self._memory.sample(batch_size, vectorizing_fn=self._minibatch_to_tf)
        states, actions, rewards, next_states, dones = memories  # (s, a, r, s', d)
        assert rewards.shape == (batch_size, 1), f"expected rewards with shape (batch, 1), received: {rewards.shape}"
        assert dones.shape == (batch_size, 1), f"expected rewards with shape (batch, 1), received: {dones.shape}"

        # Compute targets
        pi_target_out = self._ac_target.pi(next_states)
        act = pi_target_out.action
        c_target_out = self._ac_target.critic((next_states, act))  # Q_targ(s', pi_targ(s'))
        q_preds = c_target_out.critic_values
        targets = rewards + self.gamma * (1 - dones) * q_preds

        # Gradient descent step for Q function on MSBE loss
        with tf.GradientTape(watch_accessed_variables=False) as q_tape:
            q_tape.watch(self._ac.critic.trainable_variables)
            q_values = self._ac.critic((states, actions)).critic_values
            q_loss = (tf.stop_gradient(targets) - q_values) ** 2
            q_loss = tf.reduce_mean(q_loss)
        assert not tf.math.is_inf(q_loss) and not tf.math.is_nan(q_loss)
        critic_grads = q_tape.gradient(q_loss, self._ac.critic.trainable_variables)
        if self._gradient_clip_norm is not None:
            critic_grads, critic_norm = tf.clip_by_global_norm(critic_grads, self._gradient_clip_norm)
        critic_grads_and_vars = list(zip(critic_grads, self._ac.critic.trainable_variables))
        self._critic_opt.apply_gradients(critic_grads_and_vars)

        # Gradient ascent step for policy
        with tf.GradientTape(watch_accessed_variables=False) as pi_tape:
            pi_tape.watch(self._ac.pi.trainable_variables)
            pi_preds = self._ac.pi(states).action
            pi_loss = - tf.reduce_mean(self._ac.critic((states, pi_preds)).critic_values)
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
        self._train_step += 1

        if self.is_logging:
            if self._log_gradients:
                pi_grads_log = {f'actor/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                   for grad, var in pi_grads_and_vars}
                critic_grads_log = {f'critic/{".".join(var.name.split("/")[1:])}': grad.numpy()
                                    for grad, var in critic_grads_and_vars}
                if self._gradient_clip_norm is not None:
                    pi_grads_log['actor/norm'] = pi_norm
                    critic_grads_log['critic/norm'] = critic_norm
                self._log(do_log_step=False, prefix='gradients', **pi_grads_log, **critic_grads_log)
            self._log(do_log_step=False, prefix='debug', state_values=q_values.numpy(), td_targets=targets.numpy())
            self._log(do_log_step=True, policy_loss=float(pi_loss), critic_loss=float(q_loss),
                      train_step_pi=self._train_step, train_step_v=self._train_step)

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
