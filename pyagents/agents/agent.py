import abc
import tensorflow as tf
import wandb


class Agent(tf.Module, abc.ABC):
    """Abstract base class for all implemented agents.
    The agent serves two purposes:
    * Training by reading minibatches of memories, and updating an internal
      neural network (train method)
    * TODO expose policy to interact with environment

    Main methods:
    * initialize: perform initialization before training
    * train: read minibatch of memories from a replay buffer and updates
      some internal network
    """

    eps = 1e-8  # numerical stability

    def __init__(self, state_shape, action_shape, training, name='Agent'):
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
    def state_shape(self):
        return self._state_shape

    @property
    def action_shape(self):
        return self._action_shape

    @property
    def policy(self):
        return self._policy

    @property
    def config(self):
        return self._config

    def toggle_training(self, training=None):
        self._training = not self._training if training is None else training

    def _init_logger(self, wandb_params):
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

    def act(self, state, mask=None):
        """Returns the best action in the state according to current policy"""
        return self._policy.act(state, mask=mask, training=self._training)

    def _loss(self, memories):
        """Computes loss"""
        raise NotImplementedError("Loss not implemented.")

    def _log(self, do_log_step=True, prefix='', **kwargs):
        assert self._wandb_run, 'must init logger by passing wandb parameters at construction time'
        log_dict = {f'{prefix}{"/" if prefix else ""}{k}': v for k, v in kwargs.items()}
        self._log_dict.update(log_dict)
        if do_log_step:
            wandb.log(self._log_dict)
            self._log_dict = {}

    @abc.abstractmethod
    def _train(self, batch_size):
        pass

    def train(self, batch_size=None):
        return self._train(batch_size)

    @abc.abstractmethod
    def save(self, ver):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path, ver):
        pass
