import numpy as np
import tensorflow_probability as tfp
from pyagents.policies.policy import Policy


class DirichletPolicy(Policy):

    def __init__(self, state_shape, action_shape, actor_network, bounds=None):
        super().__init__(state_shape, action_shape)
        self._actor_network = actor_network
        self.bounds = bounds

    def _act(self, obs, deterministic=False, mask=None, training=True):
        alpha = self._actor_network(obs.reshape(1, *obs.shape))
        alpha = alpha.numpy().squeeze(axis=0)  # FIXME why whis squeeze?
        if deterministic:
            raise NotImplementedError()
        else:
            action = np.random.dirichlet(alpha)
        if self.bounds is not None:
            action = np.clip(action, self.bounds[0], self.bounds[1])
        return action

    def _distribution(self, obs):
        raise NotImplementedError()

    def log_prob(self, output, actions):
        dirichlet = tfp.distributions.Dirichlet(output)
        return dirichlet.log_prob(actions)
