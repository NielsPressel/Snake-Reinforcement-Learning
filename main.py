# Created: 31st of October 2019
# Author: Niels Pressel

import os
import gym
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from framework.agents.dqn import DQN, EpochalDQN, ExperimentalDQN, MinibatchDQN, Random
from framework.environments.snake_abstract import SnakeAbstract
from framework.environments.snake_simple import SnakeSimple
from framework.environments.snake import Snake
from framework.training import Training
from framework.policy import EpsilonGreedy, EpsilonAdjustmentInfo

from framework.evaluation import Evaluation


"""---Helper functions to pass into the training classes---"""


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


def create_session_info(env, model, lr, gamma, nsteps, target_network_update, memory_size, batch_size, step_count,
                        instances, rewards):
    folder_id = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    folder_path = "../weight_data/" + folder_id

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(os.path.join(folder_path, 'description.txt'), 'w') as f:
        f.write(f"Environment: {env} \nLearning rate: {lr} \nGamma: {gamma} \nnSteps: {nsteps} \n"
                f"Target network update: {target_network_update} \nMemory size: {memory_size} \nBatch size: {batch_size}"
                f" \nStep count: {step_count} \nInstance count: {instances} \nRewards: {rewards} \n\n")

        model.summary(print_fn=lambda x: f.write(x + "\n"))

    return folder_path


def chkpnt(path, agent, step):
    full_path = os.path.join(path, 'checkpoints')
    if os.path.exists(full_path):
        file_list = [f for f in os.listdir(full_path)]
        for f in file_list:
            os.remove(os.path.join(full_path, f))

    agent.save(os.path.join(full_path, f"checkpoint-{step}.dat"))


"""---Main function---"""


def main():
    tf.keras.backend.clear_session()
    tf.compat.v1.disable_eager_execution()

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    print("TensorFlow: ", tf.version.VERSION)

    if tf.config.list_physical_devices('GPU'):
        print("Using GPU version")

    # Constants
    learning_rate = 5e-3
    gamma = 0.99
    target_network_update = 10
    memory_size = 100_000
    batch_size = 1_000
    step_count = 200_000
    instance_count = 1
    n_steps = 30
    rewards = {'death': -10.0, 'food': 10.0, 'dec_distance': 0.0, 'inc_distance': 0.0}

    evaluate = False

    # Model (Neural Network to train or evaluate)
    model = Sequential(
        [
            Conv2D(16, kernel_size=(6, 6), strides=(2, 2), input_shape=(1, 22, 22), activation='relu',
                   data_format='channels_first'),
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
            Flatten(),
            Dense(256, activation='relu'),
        ]
    )

    """
    Dense(16, input_shape=(8, ), activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    """

    # Agent (object that interacts with the environment in order to learn)
    agent = DQN(model, 3, optimizer=Adam(lr=learning_rate), policy=EpsilonGreedy(1.0), mem_size=memory_size,
                target_update=target_network_update, gamma=gamma, batch_size=batch_size, nsteps=n_steps,
                policy_adjustment=EpsilonAdjustmentInfo(1.0, 0.2, 150_000, 'linear'))

    # agent = Random(3)

    # Start a new training session or evaluate an old one
    if not evaluate:
        path = create_session_info("Snake Abstract", model, learning_rate, gamma, n_steps, target_network_update,
                                   memory_size, batch_size, step_count, instance_count, rewards)
        training = Training(SnakeAbstract.create, agent)
        training.train(step_count, max_subprocesses=0, checkpnt_func=chkpnt, path=path,
                       rewards=rewards)
        agent.save(os.path.join(path, "weights.dat"), True)
        training.evaluate(10_000, visualize=True, plot_func=plot_eval)
    else:
        evaluation = Evaluation(SnakeAbstract.create, agent, "weights.dat")
        fails = evaluation.evaluate(max_rounds=20, max_steps=1_000, visualize=True, plot_func=plot_eval,
                                    step_delay=None)
        print(f"Failed {fails} times")


if __name__ == "__main__":
    main()
