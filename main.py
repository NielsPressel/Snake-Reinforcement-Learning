# Created: 31st of October 2019
# Author: Niels Pressel

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

from framework.agents.dqn import DQN
from framework.training import Training
from framework.environements.snake_abstract import SnakeAbstract


def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    for ed, steps in zip(episode_rewards, episode_steps):
        plt.plot(steps, ed)
    plt.show() if done else plt.pause(0.001)  # Pause a bit so that the graph is updated


def main():
    print("Tensorflow: ", tf.__version__)

    model = Sequential(
        [
            Flatten(input_shape=(20, 20, 4)),
            Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu")
        ]
    )

    agent = DQN(model, 4, nsteps=2)
    training = Training(SnakeAbstract.create, agent)
    training.train(100_000, 10, visualize=True, plot_func=plot_rewards)


if __name__ == "__main__":
    main()
