import gin
import numpy as np
import tensorflow_probability as tfp

from pyagents.policies.policy import Policy


@gin.configurable
class DirichletPolicy(Policy):

    def __init__(self,
                 state_shape,
                 action_shape,
                 policy_network,
                 scaling_factor: float = 1.0,
                 bounds: tuple = (-np.inf, np.inf)):
        super().__init__(state_shape, action_shape)
        self._policy_network = policy_network
        self.scaling_factor = scaling_factor
        self._bounds = bounds

    @property
    def is_discrete(self):
        return False

    @property
    def bounds(self):
        return self._bounds

    def _act(self, obs, deterministic=False, mask=None, training=True):
        alpha = self._policy_network(obs.reshape(1, *obs.shape)).dist_params
        alpha = alpha.numpy()
        if deterministic:
            raise NotImplementedError()
        else:
            action = np.array([np.random.dirichlet(alpha[0])])  # to keep batch dim
        action *= self.scaling_factor
        if self._bounds is not None:
            action = np.clip(action, self._bounds[0], self._bounds[1])
        return action

    def _distribution(self, obs):
        raise NotImplementedError()

    def entropy(self, output):
        dirichlet = tfp.distributions.Dirichlet(output)
        return dirichlet.entropy()

    def log_prob(self, output, actions):
        dirichlet = tfp.distributions.Dirichlet(output)
        return dirichlet.log_prob(actions)
