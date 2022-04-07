import abc
from typing import Optional

import numpy as np
import tensorflow as tf
import wandb

from pyagents.policies import Policy


class Agent(tf.Module, abc.ABC):
    """Abstract base class for all implemented agents.
    The agent serves two purposes:
    * Training by reading minibatches of memories (either from a buffer or an internal memory),
      and updating an internal neural network (train() method)
    * Expose policy to interact with environment (get_policy() method)
    """

    eps = 1e-8  # numerical stability

    def __init__(self, state_shape: tuple, action_shape: tuple, training: bool = True, name: str = 'Agent'):
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

    def _init_logger(self, wandb_params: dict) -> None:
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
            reinit=True, config=self.config, tags=wandb_params['tags'])
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

    @abc.abstractmethod
    def save(self, ver):
        """Saves agent to storage."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str, ver):
        """Loads agent from storage."""
        pass
