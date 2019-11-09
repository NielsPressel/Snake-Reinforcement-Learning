# Created: 8th of November 2019
# Author: Niels Pressel

import math


class EpsilonGreedyStrategy:

    def __init__(self, start, end, decay):
        self.__start = start
        self.__end = end
        self.__decay = decay

    def get_exploration_rate(self, time_step):
        return self.__end + (self.__start - self.__end) * math.exp(-1. * time_step * self.__decay)