from copy import deepcopy
from typing import Optional

import numpy as np
import tensorflow as tf
import wandb
from pyagents.agents import Agent
from pyagents.memory import Buffer, UniformBuffer
from pyagents.networks import PolicyNetwork, QNetwork
from pyagents.policies import Policy
from pyagents.utils import types


@gin.configurable
class SAC(Agent):
    """ An agent implementing the SAC algorithm """

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor: PolicyNetwork,
                 critic: QNetwork,
                 opt_actor: tf.keras.optimizers.Optimizer,
                 opt_critic: tf.keras.optimizers.Optimizer,
                 opt_alpha: tf.keras.optimizers.Optimizer,
                 critic2: Optional[QNetwork] = None,
                 policy: Policy = None,
                 buffer: Optional[Buffer] = None,
                 gamma: types.Float = 0.5,
                 standardize: bool = True,
                 actor_coef: types.Float = 1.0,
                 critic_coef: types.Float = 0.5,
                 entropy_coef: types.Float = 1.0,
                 initial_log_alpha: types.Float = 0.0,
                 target_entropy: Optional[types.Float] = None,
                 gradient_clip_norm: Optional[types.Float] = 0.5,
                 training: bool = True,
                 log_dict: dict = None,
                 name: str = 'A2C',
                 save_dir: str = './output',
                 wandb_params: Optional[dict] = None):
        super(SAC, self).__init__(state_shape, action_shape, training=training, save_dir=save_dir, name=name)

        self._actor = actor
        self._critic1 = critic
        self._critic2 = critic2
        if self._critic2 is None:
            self._critic2 = deepcopy(self._critic1)
        self._opt_actor = opt_actor
        self._opt_critic = opt_critic
        self._opt_alpha = opt_alpha
        if policy is None:
            self._policy = self._actor.get_policy()
        else:
            self._policy = policy
        if buffer is not None:
            buffer.set_save_dir(self._save_dir)
            self._memory: Buffer = buffer
        else:
            self._memory: Buffer = UniformBuffer(save_dir=self._save_dir)
        self.gamma = gamma
        self._standardize = standardize
        self._actor_coef = actor_coef
        self._critic_coef = critic_coef
        self._entropy_coef = entropy_coef
        self._gradient_clip_norm = gradient_clip_norm
        self.target_entropy = target_entropy if target_entropy else -np.prod(self.action_shape)/2.0
        self._initial_log_alpha = initial_log_alpha

        # update config

        if log_dict is not None:
            self.config.update(**log_dict)

        if wandb_params:
            self._init_logger(wandb_params)

    def _wandb_define_metrics(self):
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")
        wandb.define_metric('entropy_loss', step_metric="train_step", summary="min")
