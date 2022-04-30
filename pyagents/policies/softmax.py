import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pyagents.policies.policy import Policy, PolicyOutput


class SoftmaxPolicy(Policy):

    @property
    def is_discrete(self) -> bool:
        return True

    def __init__(self, state_shape, action_shape, policy_network):
        super().__init__(state_shape, action_shape)
        self._policy_network = policy_network

    def _act(self, obs, deterministic=False, mask=None, training=True) -> PolicyOutput:
        pi_out = self._policy_network(obs)
        probs = pi_out.dist_params
        probs = probs.numpy()
        dist = tfp.distributions.Categorical(probs=probs)
        if deterministic:
            act = np.argmax(probs, axis=0)
        else:
            act = dist.sample().numpy()
        lp = dist.log_prob(act).numpy()
        return PolicyOutput(actions=act, logprobs=lp)

    def entropy(self, output):
        dist = tfp.distributions.Categorical(probs=output)
        return dist.entropy()

    def log_prob(self, output, actions):
        dist = tfp.distributions.Categorical(probs=output)
        return dist.log_prob(tf.squeeze(actions))

    def _distribution(self, obs):
        raise NotImplementedError()
