from framework.core import Memory, Transition
from collections import deque, namedtuple
import random
import numpy as np

"""---Memory helper function---"""


def unpack(traces):
    """Returns states, actions, rewards, end_states, and a mask for episode boundaries given traces."""
    states = [t[0].state for t in traces]
    actions = [t[0].action for t in traces]
    rewards = [[e.reward for e in t] for t in traces]
    end_states = [t[-1].next_state for t in traces]
    not_done_mask = [[1 if n.next_state is not None else 0 for n in t] for t in traces]
    return states, actions, rewards, end_states, not_done_mask


"""---ReplayMemory class---"""


class ReplayMemory(Memory):
    """This class implements a simple experience storing memory.

    Attributes:
        traces (deque): Buffer to store the traces with maxlen of capacity
        buffer (array): Buffer to store a concrete number (steps) of transitions
        capacity (int): Maximum number of traces stored in this memory
        steps (int): Number of steps after that a new trace is created
        exclude_boundaries (bool): Determines if the buffer gets cleared on every terminal state
    """

    def __init__(self, capacity, steps=1, exclude_boundaries=False):
        self.traces = deque(maxlen=capacity)
        self.capacity = capacity

    def put(self, transition):
        self.traces.append(transition)

    def get(self, batch_size):
        traces = random.sample(self.traces, batch_size)

        state_batch = [t.state for t in traces]
        action_batch = [t.action for t in traces]
        reward_batch = [t.reward for t in traces]
        next_state_batch = [t.next_state for t in traces]
        return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), None

    def clear(self):
        self.traces.clear()

    def __len__(self):
        """Returns length of trace buffer."""
        return len(self.traces)


"""---PrioritizedExperienceReplay class---"""

"""
The following part is copied from huskarl framework! (https://github.com/danaugrs/huskarl)
"""

EPS = 1e-3  # Constant added to all priorities to prevent them from being zero


class PrioritizedExperienceReplay(Memory):
    """Stores prioritized interaction with an environment in a priority queue implemented via a heap.

    Provides efficient prioritized sampling of multistep traces.
    If exclude_boundaries==True, then traces are sampled such that they don't include episode boundaries.
    For more information see "Prioritized Experience Replay" (Schaul et al., 2016).
    """

    def __init__(self, capacity, steps=1, exclude_boundaries=False, prob_alpha=0.6):
        """
        Args:
            capacity (int): The maximum number of traces the memory should be able to store.
            steps (int): The number of steps (transitions) each sampled trace should include.
            exclude_boundaries (bool): If True, sampled traces will not include episode boundaries.
            prob_alpha (float): Value between 0 and 1 that specifies how strongly priorities are taken into account.
        """
        self.traces = []  # Each element is a tuple containing self.steps transitions
        self.priorities = np.array([])  # Each element is the priority for the same-index trace in self.traces
        self.buffer = []  # Rolling buffer of size at most self.steps
        self.capacity = capacity
        self.steps = steps
        self.exclude_boundaries = exclude_boundaries
        self.prob_alpha = prob_alpha
        self.traces_idxs = []  # Temporary list that contains the indexes associated to the last retrieved traces

    def put(self, transition):
        """Adds transition to memory."""
        # Append transition to temporary rolling buffer
        self.buffer.append(transition)
        # If buffer doesn't yet contain a full trace - return
        if len(self.buffer) < self.steps: return
        # If self.traces not at max capacity, append new trace and priority (use highest existing priority if available)
        if len(self.traces) < self.capacity:
            self.traces.append(tuple(self.buffer))
            self.priorities = np.append(self.priorities, EPS if self.priorities.size == 0 else self.priorities.max())
        else:
            # If self.traces at max capacity, substitute lowest priority trace and use highest existing priority
            idx = np.argmin(self.priorities)
            self.traces[idx] = tuple(self.buffer)
            self.priorities[idx] = self.priorities.max()
        # If excluding boundaries and we've reached a boundary - clear the buffer
        if self.exclude_boundaries and transition.next_state is None:
            self.buffer = []
            return
        # Roll buffer
        self.buffer = self.buffer[1:]

    def get(self, batch_size):
        """Samples the specified number of traces from the buffer according to the prioritization and prob_alpha."""
        # Transform priorities into probabilities using self.prob_alpha
        probs = self.priorities ** self.prob_alpha
        probs /= probs.sum()
        # Sample batch_size traces according to probabilities and store indexes
        self.traces_idxs = np.random.choice(len(self.traces), batch_size, p=probs, replace=False)
        traces = [self.traces[idx] for idx in self.traces_idxs]
        return unpack(traces)

    def clear(self):
        self.traces.clear()
        self.buffer.clear()
        self.priorities = np.array([])
        self.traces_idxs = []

    def last_traces_idxs(self):
        """Returns the indexes associated with the last retrieved traces."""
        return self.traces_idxs.copy()

    def update_priorities(self, traces_idxs, new_priorities):
        """Updates the priorities of the traces with specified indexes."""
        self.priorities[traces_idxs] = new_priorities + EPS

    def __len__(self):
        """Returns the number of traces stored."""
        return len(self.traces)


ProbabilityAdjustment = namedtuple('ProbabilityAdjustment', ['prob_start', 'prob_end', 'step_count',
                                                             'interpolation_type'])


class DualExperienceReplay(Memory):

    def __init__(self, capacity, steps=1, prob=0.8, adjustment=None):
        self.capacity_m1 = capacity // 2
        self.capacity_m2 = capacity // 2
        self.steps = steps
        self.last_array_ops = []
        self.traces_m1 = []
        self.traces_m2 = []
        self.prob = prob
        self.adjustment = adjustment
        self.performance_padding = 1_000

    def put(self, transition):
        self.last_array_ops.append(id(transition))

        if len(self.last_array_ops) > 40:
            self.last_array_ops.pop(0)

        if transition.reward >= 0.5:
            self.traces_m1.append(transition)

            if len(self.traces_m1) > self.capacity_m1 + self.performance_padding:
                self.traces_m1 = self.traces_m1[self.performance_padding:]
        else:
            self.traces_m2.append(transition)

            if len(self.traces_m2) > self.capacity_m2 + self.performance_padding:
                self.traces_m2 = self.traces_m2[self.performance_padding:]

    def get(self, batch_size):
        m1_size = int(np.ceil(self.prob * batch_size))
        m2_size = int(np.floor((1.0 - self.prob) * batch_size))

        indices_m1 = np.random.choice(len(self.traces_m1), m1_size, replace=False)
        indices_m2 = np.random.choice(len(self.traces_m2), m2_size, replace=False)

        m1_batch = [self.traces_m1[i] for i in indices_m1]
        m2_batch = [self.traces_m2[i] for i in indices_m2]

        batch = m1_batch + m2_batch

        state_batch = [t.state for t in batch]
        action_batch = [t.action for t in batch]
        reward_batch = [t.reward for t in batch]
        next_state_batch = [t.next_state for t in batch]
        return np.array(state_batch), np.array(action_batch), np.array(reward_batch), np.array(next_state_batch), None

    def clear(self):
        self.traces_m1 = []
        self.traces_m2 = []

    def adjust_prob(self, step):
        if self.adjustment:
            if step <= self.adjustment.step_count:
                if self.adjustment.interpolation_type == 'linear':
                    self.prob = ((self.adjustment.prob_end - self.adjustment.prob_start) / self.adjustment.step_count) \
                                * step + self.adjustment.prob_start

    def adjust_rewards(self, reward, steps):
        steps = int(min(steps, 40))

        for operation in reversed(self.last_array_ops[-steps:]):
            found = False
            for item in reversed(self.traces_m1[-steps:]):
                if operation == id(item):
                    self.traces_m1.remove(item)
                    transition = item
                    transition.adjust_reward(reward)

                    self.traces_m2.append(transition)
                    found = True
                    break

            if not found:
                for item in reversed(self.traces_m2[-steps:]):
                    if operation == id(item):
                        item.adjust_reward(reward)
                        break

    def __len__(self):
        return min(len(self.traces_m1), len(self.traces_m2))