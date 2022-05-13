import numpy as np


class SumTree:
    """A sum tree used for efficiently storing and sampling prioritized memories.
       A sum tree is a complete binary tree where leaves contain priorities and inner nodes
       contain the sum of their subtrees."""

    def __init__(self, size: int):
        """Creates the sum tree data structure.

        Args:
            size: total number of nodes maintaned in the tree.
        """

        assert isinstance(size, int) and size >= 0, f'wrong sum tree size {size}'
        self._nodes = []
        tree_depth = int(np.ceil(np.log2(size))) + 1
        for nodes_depth_d in (2 ** d for d in range(tree_depth)):  # level at depth d contains 2 ** d nodes
            self._nodes.append(np.zeros(nodes_depth_d))
        self._max_p = 1.0

    def __len__(self):
        return np.sum(self._nodes[-1] != 0.)

    def total_priority(self):
        return self._nodes[0][0]

    def sample(self, lb=0.0, ub=1.0):
        """Samples a node from the sum tree."""
        value = np.random.uniform(lb, ub) * self._nodes[0][0]  # sample a random number in (lb, ub]
        idx = 0
        for nodes_depth_d in self._nodes[1:]:
            left = idx * 2  # index of left child
            if value < nodes_depth_d[left]:  # recurse into left subtree
                idx = left
            else:
                idx = left + 1  # recurse into right subtree
                value -= nodes_depth_d[left]  # update value to match new subtree range
        return idx

    def get(self, idx: int):
        """Returns the priority of a leaf node."""
        return self._nodes[-1][idx]

    def set(self, idx: int, value: float = None):
        """Sets value of a node in the tree and updates its parents.

        Args:
            idx: index of the node to update
            value: new priority value to store
        """
        if value is None:
            value = self._max_p
        else:
            assert value > 0.0, f'sum tree cannot hold negative values, received: {value}'
            self._max_p = max(value, self._max_p)

        delta = value - self._nodes[-1][idx]
        # starting from this node, traverse the tree up to the root updating intermediate nodes
        for nodes_depth_d in self._nodes[::-1]:
            nodes_depth_d[idx] += delta
            idx //= 2  # compute index of parent

