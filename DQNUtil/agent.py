# Created: 8th of November 2019
# Author: Niels Pressel

import os
import random
import numpy as np
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from DQNUtil.experience import Experience


class Agent:

    def __init__(self, strategy, width, height, frame_count, lr, action_count):
        self.__time_step = 0
        self.__strategy = strategy
        self.__action_count = action_count

        self._create_networks(width, height, frame_count, lr)

        self.__policy_network.summary()
        self.__target_network.summary()

    def select_action(self, state):
        rate = self.__strategy.get_exploration_rate(self.__time_step)
        self.__time_step += 1

        if rate > random.random():
            return random.randrange(self.__action_count)
        else:
            return np.argmax(self.__policy_network.predict(state)[0])

    def _create_networks(self, width, height, frame_count, lr):
        self.__policy_network = self._build_compile_model(width, height, frame_count, lr)
        self.__target_network = self._build_compile_model(width, height, frame_count, lr)
        self.align_target_network()

    def _build_compile_model(self, width, height, frame_count, lr):
        model = Sequential()
        model.add(Dense(128, activation="relu", input_shape=(width, height, frame_count)))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        model.add(Dense(self.__action_count, activation="linear"))

        model.compile(loss='mse', optimizer=Adam(lr=lr))
        return model

    def align_target_network(self):
        self.__target_network.set_weights(self.__policy_network.get_weights())

    def train(self, memory, batch_size, gamma):
        exp = memory.sample(batch_size)

        for state, action, reward, next_state in exp:
            state = np.reshape(state, [1, 20, 20, 3])
            next_state = np.reshape(next_state, [1, 20, 20, 3])

            target = self.__policy_network.predict(state)
            t = self.__target_network.predict(next_state)
            target[0][action] = reward + gamma * np.amax(t)

            self.__policy_network.fit(state, target, epochs=1, verbose=0)

    def save_model(self, session_name, episode):
        path = '../data/training/models/' + session_name
        if not os.path.exists(path):
            os.makedirs(path)
        self.__policy_network.save(path + '/policy_network_ep_' + str(episode) + '.h5')
