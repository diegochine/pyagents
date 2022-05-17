from copy import deepcopy
from typing import Optional, Dict, List
import gin
import numpy as np
import tensorflow as tf
import wandb
from pyagents.agents import Agent
from pyagents.agents.off_policy_agent import OffPolicyAgent
from pyagents.memory import Buffer, UniformBuffer, load_memories
from pyagents.networks import PolicyNetwork, DiscreteQNetwork
from pyagents.policies import Policy


@gin.configurable
class SAC(OffPolicyAgent):
    """ An agent implementing the SAC algorithm """

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 actor: PolicyNetwork,
                 critic: DiscreteQNetwork,
                 actor_opt: tf.keras.optimizers.Optimizer,
                 critic_opt: tf.keras.optimizers.Optimizer,
                 alpha_opt: tf.keras.optimizers.Optimizer,
                 critic2: Optional[DiscreteQNetwork] = None,
                 policy: Policy = None,
                 buffer: Optional[Buffer] = None,
                 gamma: float = 0.5,
                 standardize: bool = True,
                 actor_coef: float = 1.0,
                 critic_coef: float = 0.5,
                 entropy_coef: float = 1.0,
                 initial_log_alpha: float = 0.0,
                 target_entropy: Optional[float] = None,
                 gradient_clip_norm: Optional[float] = 0.5,
                 training: bool = True,
                 log_dict: dict = None,
                 name: str = 'A2C',
                 save_dir: str = './output',
                 wandb_params: Optional[dict] = None):
        super(SAC, self).__init__(state_shape,
                                  action_shape,
                                  training=training,
                                  buffer=buffer,
                                  save_dir=save_dir,
                                  name=name)
        if actor_opt is None or critic_opt is None or alpha_opt is None and training:
            raise ValueError('agent cannot be trained without optimizers')

        self._actor = actor
        self._critic1 = critic
        self._critic2 = critic2
        if self._critic2 is None:
            self._critic2 = deepcopy(self._critic1)
        self._actor_opt = actor_opt
        self._critic_opt = critic_opt
        self._alpha_opt = alpha_opt
        if policy is None:
            self._policy = self._actor.get_policy()
        else:
            self._policy = policy
        self.gamma = gamma
        self._standardize = standardize
        self._actor_coef = actor_coef
        self._critic_coef = critic_coef
        self._entropy_coef = entropy_coef
        self._gradient_clip_norm = gradient_clip_norm
        self.target_entropy = target_entropy if target_entropy else -np.prod(self.action_shape)/2.0
        self._initial_log_alpha = initial_log_alpha

        # update config

        if wandb_params:
            self._init_logger(wandb_params, config={**self.config, **log_dict})

    def _wandb_define_metrics(self):
        wandb.define_metric('policy_loss', step_metric="train_step", summary="min")
        wandb.define_metric('critic_loss', step_metric="train_step", summary="min")
        wandb.define_metric('entropy_loss', step_metric="train_step", summary="min")

    def _networks_config_and_weights(self):
        net_config = self._online_q_network.get_config()
        net_weights = self._online_q_network.get_weights()
        return [('q_net', net_config, net_weights)]

    @staticmethod
    def networks_name() -> List[tuple]:
        return [('q_net', DiscreteQNetwork)]

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
        q_net = networks['q_net']
        agent_config.update({'q_network': q_net,
                             'buffer': buffer})
        return agent_config
