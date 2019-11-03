import tensorflow as tf
from collections import deque
import numpy as np
import random
from memory_profiler import profile


class DeepQNetwork:

    EXPLORATION_MAX = 1.0
    EXPLORATION_MIN = 0.01

    def __init__(self, state_size, action_size, frame_count, learning_rate):
        tf.compat.v1.disable_eager_execution()

        self.__model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(32, input_shape=(state_size, frame_count), activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(action_size, activation="linear")
            ]
        )

        self.__model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss="mse")
        self.__model.summary()

        self.__action_size = action_size
        self.__batch_size = 20
        self.__gamma = 0.95
        self.__exploration_decay = 0.995

        self.__exploration_rate = self.EXPLORATION_MAX
        self.__memory = deque(maxlen=1000)

    def remember(self, state, action, reward, next_state, done):
        self.__memory.append((state, action, reward, next_state, done))

    def decide(self, state):
        if np.random.rand() < self.__exploration_rate:
            return random.randrange(self.__action_size)
        q_values = self.__model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.__memory) < self.__batch_size:
            return
        batch = random.sample(self.__memory, self.__batch_size)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + self.__gamma * np.amax(self.__model.predict(next_state)[0]))
            q_values = self.__model.predict(state)
            q_values[0][action] = q_update
            self.__model.fit(state, q_values, verbose=0)

    def adjust_exploration_rate(self):
        self.__exploration_rate *= self.__exploration_decay
        self.__exploration_rate = max(self.EXPLORATION_MIN, self.__exploration_rate)
        print(self.__exploration_rate)