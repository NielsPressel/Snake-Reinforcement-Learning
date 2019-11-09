# Created: 8th of November 2019
# Author: Niels Pressel

import random


class ReplayMemory:

    def __init__(self, capacity):
        self.__capacity = capacity
        self.__memory = []
        self.__push_count = 0

    def push(self, experience):
        if len(self.__memory) < self.__capacity:
            self.__memory.append(experience)
        else:
            self.__memory[self.__push_count % self.__capacity] = experience
        self.__push_count += 1

    def sample(self, batch_size):
        return random.sample(self.__memory, batch_size)

    def is_sample_available(self, batch_size):
        return len(self.__memory) >= batch_size
