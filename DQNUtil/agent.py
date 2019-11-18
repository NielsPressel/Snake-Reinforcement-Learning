# Created: 8th of November 2019
# Author: Niels Pressel

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation
from tensorflow.keras.optimizers import Adam
from DQNUtil.replay_memory import ReplayMemory


def build_dqn(lr, input_dims, number_actions, fc1_dims):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_dims,), activation='relu'))
    model.add(Dense(fc1_dims, activation='relu'))
    model.add(Dense(number_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')
    model.summary()
    return model


class Agent:

    def __init__(self, alpha, gamma, number_actions, epsilon, batch_size, replace, input_dims, eps_dec=0.9967,
                 eps_min=0.01, mem_size=100000):

        self.action_space = [i for i in range(number_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.replace = replace
        self.learn_step = 0
        self.memory = ReplayMemory(mem_size, input_dims)
        self.q_eval = build_dqn(alpha, input_dims, number_actions, 512)
        self.q_target = build_dqn(alpha, input_dims, number_actions, 512)

    def align_target_network(self):
        if self.replace is not None and self.learn_step % self.replace == 0:
            self.q_eval.set_weights(self.q_target.get_weights())

    def select_action(self, observation, use_epsilon=True):
        if np.random.random() < self.epsilon and use_epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation], copy=False, dtype=np.float32)
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def train(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, next_state, terminal = self.memory.sample(self.batch_size)

            self.align_target_network()

            q_eval = self.q_eval.predict(state)
            q_next = self.q_target.predict(next_state)

            q_target = q_eval[:]

            indices = np.arange(self.batch_size)

            q_target[indices, action] = reward + self.gamma * np.max(q_next, axis=1) * terminal

            self.q_eval.train_on_batch(state, q_target)

            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
            self.learn_step += 1

    def push_observation(self, state, action, reward, next_state, terminal):
        self.memory.push(state, action, reward, next_state, terminal)

    def save_model(self):
        print("---Saving models---")
        self.q_eval.save("eval.h5")
        self.q_target.save("target.h5")

    def load_models(self):
        print("--Loading Models---")
        self.q_eval = load_model("eval.h5")
        self.q_target = load_model("target.h5")