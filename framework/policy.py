import random
import numpy as np
from collections import namedtuple

from framework.core import Policy

"""---EpsilonAdjustmentInfo namedtuple---

Use this to define the epsilon decay while training.
"""


EpsilonAdjustmentInfo = namedtuple('EpsilonAdjustmentInfo', ['epsilon_start', 'epsilon_end', 'step_count',
                                                             'interpolation_type'])

"""---EpsilonGreedy class---"""


class EpsilonGreedy(Policy):
    """Implementation of a simple learning policy.

    Based on epsilon the policy picks a random action or the action with largest value.
    probability: epsilon -> random, 1 - epsilon -> largest value

    Attributes:
        epsilon (float): Determines whether the action with the largest value or a random action is chosen.
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def act(self, q_values):
        """Chooses an action based on given QValues.

        Args:
            q_values (np.ndarray): QValues to calculate the largest value from.

        Returns:
            int: The action that was chosen.
        """
        if random.random() > self.epsilon:
            return np.argmax(q_values)
        return random.randrange(len(q_values))


"""---Greedy class---"""


class Greedy(Policy):
    """Implementation of a simple testing policy.

    This policy just returns the index of the largest value.
    """

    def act(self, q_values):
        """Chooses the index with the largest value.

        Args:
            q_values (np.ndarray): QValues to calculate the largest value from.

        Returns:
            int: The index chosen.
        """
        return np.argmax(q_values)
