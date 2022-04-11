import abc
import json
import os
from typing import Optional, List, Dict

import h5py
import numpy as np
import tensorflow as tf
import wandb

from pyagents.policies import Policy
from pyagents.utils import json_utils


def update_target(source_vars, target_vars, tau=1.0):
    # source_variables = self._online_q_network.variables
    # target_variables = self._target_q_network.variables
    for (sv, tv) in zip(source_vars, target_vars):
        tv.assign((1 - tau) * tv + tau * sv)


class Agent(tf.Module, abc.ABC):
    """Abstract base class for all implemented agents.
    The agent serves two purposes:
    * Training by reading minibatches of memories (either from a buffer or an internal memory),
      and updating an internal neural network (train() method)
    * Expose policy to interact with environment (get_policy() method)
    """

    eps = 1e-8  # numerical stability

    AGENT_FILE = 'agent'

    def __init__(self,
                 state_shape: tuple,
                 action_shape: tuple,
                 training: bool,
                 save_dir: str = './output',
                 save_memories: bool = False,
                 name='Agent'):
        """Creates an Agent.

        Args:
            state_shape: Tuple representing the shape of the input space (observations).
            action_shape: Tuple representing the shape of the action space.
            training: (Optional) If True, agent is in training phase. Defaults to True.
            name: (Optional) Name of the agent.
        """
        super(Agent, self).__init__(name=name)
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._training = training
        self._save_memories = save_memories
        self._save_dir = os.path.join(save_dir, name)
        if not os.path.isdir(self._save_dir):
            os.makedirs(self._save_dir)
        self._wandb_run = None
        self._log_dict = None
        self.is_logging = False
        self._config = {'name': name}
        self._train_step = 0

    @property
    def state_shape(self) -> tuple:
        return self._state_shape

    @property
    def action_shape(self) -> tuple:
        return self._action_shape

    @property
    def config(self) -> dict:
        return self._config

    def get_policy(self) -> Policy:
        return self._policy

    def toggle_training(self, training: Optional[bool] = None) -> None:
        """Allows toggling between training and evaluation mode.

        Args:
            training: (Optional) If True, set agent to training mode; if False, set agent to evaluation mode;
              if None, toggles current mode. Defaults to None.
        """
        self._training = not self._training if training is None else training

    def _init_logger(self, wandb_params: dict, config: dict) -> None:
        """Initializes WandB logger.

         Args:
             wandb_params: wandb configuration dictionary with the following keys:
                * project: str, name of WandB project
                * entity: str, WandB entity that owns the project
                * group: None or str, WandB group of the run
                * tags: None or list of str, optional list of tags
         """
        wandb.login(key=wandb_params['key'])
        self._wandb_run = wandb.init(
            project=wandb_params['project'], entity=wandb_params['entity'], group=wandb_params['group'],
            reinit=True, config=config, tags=wandb_params['tags'])
        wandb.define_metric('train_step', summary='max')
        self._wandb_define_metrics()
        self._log_dict = {}
        self.is_logging = True

    def _initialize(self):
        pass

    def initialize(self):
        pass

    def act(self, state: np.ndarray, mask: Optional[np.ndarray] = None):
        """Returns the best action in the state according to current policy.

        Args:
            state: Current state.
            mask: (Optional) Boolean mask for illegal actions. Defaults to None
        """
        return self._policy.act(state, mask=mask, training=self._training)

    def _loss(self, memories):
        """Computes loss."""
        raise NotImplementedError("Loss not implemented.")

    def _log(self, do_log_step: bool = True, prefix: str = '', **kwargs) -> None:
        """Logs all kwargs to WandB.

        Args:
            do_log_step: (Optional) If False, caches data and waits next _log() call to perform actual logging.
              Defaults to True.
            prefix: (Optional) A str that is prefixed to all keys to be logged.
              Defaults to "" (empty string, no prefix).
        """
        assert self._wandb_run, 'must init logger by passing wandb parameters at construction time'
        log_dict = {f'{prefix}{"/" if prefix else ""}{k}': v for k, v in kwargs.items()}
        self._log_dict.update(log_dict)
        if do_log_step:
            wandb.log(self._log_dict)
            self._log_dict = {}

    @abc.abstractmethod
    def _train(self, batch_size: Optional[int], *args, **kwargs) -> dict:
        """Performs training step.

        Subclasses must implement this method.
        Args:
            batch_size: (Optional) batch size for this training step.
            Defaults to None for on-policy methods, and to 128 for off-policy methods.
        Returns:
            dict with loss names (str) as keys and corresponding values (float) as values.
        """
        pass

    def train(self, batch_size: Optional[int] = None, *args, **kwargs):
        """Performs training step, and eventually logs loss values.

        Subclasses must not override this method, but must implement _train().
        Args:
            batch_size: (Optional)
        Returns:
            dict of losses
        """
        losses_dict = self._train(batch_size, *args, **kwargs)
        if self._wandb_run is not None:
            self._log(**losses_dict)
        return losses_dict

    @abc.abstractmethod
    def remember(self, state: np.ndarray, action, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Stores piece of experience in internal memory/buffer.

        Subclasses are expected to implement this method.
        Args:
            state: current state s_t
            action: action a_t executed by the agent, according to current policy
            reward: reward r_(t+1) obtained by the agent following the execution of a_t in s_t
            next_state: state s_(t+1)
        """
        pass

    def _wandb_define_metrics(self):
        """Defines WandB metrics.

        Subclasses must implement this function to enable WandB logging.
        """
        raise NotImplementedError('Must implement _wandb_define_metrics method to enable WandB logging.')

    def _do_save_memories(self):
        pass

    @abc.abstractmethod
    def _networks_config_and_weights(self) -> List[tuple]:
        """ Returns a list of tuples (name, config, weights)
        for each network that the agents wants to be saved"""
        pass

    def save(self, ver):
        """Saves agent to storage."""
        if self._save_memories:
            self._do_save_memories()

        f = h5py.File(f'{self._save_dir}/{self.AGENT_FILE}_v{ver}', mode='w')
        try:
            agent_group = f.create_group('agent')
            for k, v in self._config.items():
                agent_group.attrs[k] = v

            for net_name, net_config, net_weights in self._networks_config_and_weights():
                net_config_group = f.create_group(f'{net_name}_config')
                for k, v in net_config.items():
                    if isinstance(v, (dict, list, tuple)):  # FIXME sto dumps non funziona dio cane
                        v = json.dumps(v, default=json_utils.get_json_type).encode('utf8')
                        net_config_group.attrs.create(k, np.void(v))
                    else:
                        net_config_group.attrs[k] = v

                net_weights_group = f.create_group(f'{net_name}_weights')
                for i, lay_weights in enumerate(net_weights):
                    net_weights_group.create_dataset(f'{net_name}_weights{i:0>3}', data=lay_weights)

            f.flush()
        finally:
            f.close()

    @staticmethod
    def networks_name() -> List[tuple]:
        """ Returns a list of tuples (name, class)
            for each network that needs to be loaded from storage."""
        raise NotImplementedError('Must implement networks_name method to enable WandB logging.')

    @staticmethod
    def generate_input_config(
            agent_config: dict,
            networks: dict,
            optimizer: tf.keras.optimizers,
            load_mem: bool,
            path: str) -> Dict:
        """ Returns a dictionary of the input parameters in order to
            instantiate the child class."""
        raise NotImplementedError('Must implement generate_input_config method to enable WandB logging.')

    @classmethod
    def load(cls, path: str, ver, optimizer=None, load_mem: bool = False, **kwargs):
        """Loads agent from storage."""
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()

        f = h5py.File(f'{path}/{cls.AGENT_FILE}_v{ver}', mode='r')
        agent_group = f['agent']
        agent_config = {}
        for k, v in agent_group.attrs.items():
            agent_config[k] = v

        networks = {}
        for net_name, net_class in cls.networks_name():
            net_config = {}
            net_config_group = f[f'{net_name}_config']
            for i in net_config_group.attrs.items():
                # if hasattr(i, 'decode'):
                #     i = i.decode('utf-8')
                # i = json_utils.decode(i)
                k,v = i
                if isinstance(v, np.void):
                    net_config[k] = json_utils.decode(v.tobytes())
                else:
                    if v == '[]':
                        v = None
                    net_config[k] = v
            net_weights_group = f[f'{net_name}_weights']
            net_weights_group = {name: weights[...] for name, weights in net_weights_group.items()}
            net_weights = [net_weights_group[f'{net_name}_weights{i:0>3}'] for i in range(len(net_weights_group))]
            net_config['trainable'] = trainable
            net = net_class.from_config(net_config)
            # FIXME choose another name like input_shape (safer) or let child assign weights
            net(tf.ones((1, *net_config['state_shape'])))
            net.set_weights(net_weights)
            networks[net_name] = net
        f.close()

        input_dict = cls.generate_input_config(agent_config, networks, optimizer, load_mem, path)

        input_dict.update(kwargs)
        return cls(**input_dict)
