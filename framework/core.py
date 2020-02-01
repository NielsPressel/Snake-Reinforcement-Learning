"""This file implements all the base classes for the framework types.

Use these classes for subclassing.

Typical usage example:

    class DDQNAgent(Agent):

        def __init__(self):
            pass
"""

from collections import namedtuple


class Agent:
    """Abstract base class for agents. Do not instantiate this!"""

    def act(self, state):
        """This method calculates the action to take from a given state.

        Calculates the action by passing the state into the evaluation network. The resulting q_values may be
        manipulated by the agent's policy. The policy's result will be returned.

        Args:
            state (np.ndarray): State the agent observed in the last step it took in the given environment.

        Returns:
            int: The action to take, this is calculated by the evaluation network and the policy.
        """
        raise NotImplementedError()

    def push_observation(self, transition):
        """This method pushes the observed transition into replay memory for training.

        Args:
            transition (np.ndarray): State the agent observed in the last step it took in the given environment.
        """
        raise NotImplementedError()

    def train(self, step):
        """Trains the agent based on its experiences stored in replay memory."""
        raise NotImplementedError()

    def save(self, filename):
        """This method saves the agent's training progress.

        Args:
            filename: String that points to the location to save the agent's model to.
        """
        raise NotImplementedError()

    def load(self, filename):
        raise NotImplementedError()


class Policy:
    """Abstract base class for policies. Do not instantiate this!"""

    def act(self, q_values):
        """This method manipulates the given q_values according to the policy.

        Args:
            q_values (np.ndarray): QValues that were calculated by your neural network.

        Returns:
            object (int or np.ndarray): Calculation result.
        """
        raise NotImplementedError()


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class Memory:
    """Abstract base class for memories. Do not instantiate this!

    A Memory saves environment transitions it observed earlier to learn from these.
    """

    def put(self, *args):
        raise NotImplementedError()

    def get(self, *args):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class Environment:
    """Abstract base class for environment building. Do not instantiate this!"""

    @classmethod
    def create(cls, *args):
        raise NotImplementedError()

    def step(self, action):
        """This method executes on step in the environment based on a given action.

        Args:
            action: The action chosen by the agent.

        Returns:
            (float, np.ndarray, boolean): Returns a tuple of reward, next_state, done.

        """
        raise NotImplementedError()

    def render(self):
        """This method renders the environment.

        You must call this after every call to step for smooth rendering.
        """
        raise NotImplementedError()

    def reset(self):
        """This method resets the environment to its originally state.

        You also have to call this method after instantiating your environment.

        Returns:
            np.ndarray: The initial state.
        """
        raise NotImplementedError()
