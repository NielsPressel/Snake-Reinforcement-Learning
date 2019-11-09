# Created: 8th of November 2019
# Author: Niels Pressel

import random
import numpy as np


class Agent:

    def __init__(self, strategy, action_count):
        self.__time_step = 0
        self.__strategy = strategy
        self.__action_count = action_count

    def select_action(self, state, policy_network):
        rate = self.__strategy.get_exploration_rate(self.__time_step)
        self.__time_step += 1

        if rate > random.random():
            return random.randrange(self.__action_count)
        else:
            return np.argmax(policy_network.predict(state)[0])
