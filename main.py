# Created: 31st of October 2019
# Author: Niels Pressel

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

from framework.agents.dqn import DQN
from framework.environements.snake_abstract import SnakeAbstract
from framework.environements.snake_simple import SnakeSimple
from framework.training import Training

from framework.evaluation import Evaluation

import gym


def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    for ed, steps in zip(episode_rewards, episode_steps):
        plt.plot(steps, ed)
    plt.show() if done else plt.pause(0.001)  # Pause a bit so that the graph is updated


def plot_eval(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.plot(episode_steps, episode_rewards)
    plt.show() if done else plt.pause(0.001)


def main():
    print("Tensorflow: ", tf.__version__)

    model = Sequential(
        [
            Flatten(input_shape=(20, 20, 2)),
            Dense(32, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
        ]
    )

    agent = DQN(model, 4, nsteps=2)

    """
    training = Training(SnakeAbstract.create, agent)
    training.train(100_000, 20, visualize=False, plot_func=plot_rewards)
    agent.save("test_weights.dat")
    training.evaluate(10_000, visualize=True, plot_func=plot_eval)
    """

    evaluation = Evaluation(SnakeAbstract.create, agent, "C:\\Users\\niels\\Documents\\Facharbeit\\weight_data\\3-1-2020\\test_weights.dat")
    fails = evaluation.evaluate(max_rounds=20, max_steps=1_000, visualize=True, plot_func=plot_eval)
    print("Failed %d times" % fails)


if __name__ == "__main__":
    main()
