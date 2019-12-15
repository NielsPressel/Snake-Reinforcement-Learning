import random
import numpy as np
from framework.core import Policy


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
        """This method chooses an action based on given QValues.

        Args:
            q_values (np.ndarray): QValues to predict the largest value from.

        Returns:
            int: The action that was chosen.
        """
        if random.random() > self.epsilon:
            return np.argmax(q_values)
        return random.randrange(len(q_values))