# Created: 31st of October 2019
# Author: Niels Pressel

import os
import gym
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import numpy as np

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from framework.agents.dqn import DQN, EpochalDQN, ExperimentalDQN, MinibatchDQN, Random
from framework.environments.snake_abstract import SnakeAbstract
from framework.environments.snake_simple import SnakeSimple
from framework.environments.snake_abstract_framed import SnakeAbstractFramed
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
                        instances, rewards, seed):
    folder_id = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    folder_path = "../weight_data/" + folder_id

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(os.path.join(folder_path, 'description.txt'), 'w') as f:
        f.write(f"Seed: {seed} \nEnvironment: {env} \nLearning rate: {lr} \nGamma: {gamma} \nnSteps: {nsteps} \n"
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

    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print("TensorFlow: ", tf.version.VERSION)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Using GPU version")
        tf.config.experimental.set_memory_growth(gpus[0], True)

    if tf.executing_eagerly():
        print("Executing eagerly")
    else:
        print("Eager execution is disabled")

    # Constants
    learning_rate = 3e-3
    gamma = 0.95
    target_network_update = 1_000
    memory_size = 500_000
    batch_size = 1024
    step_count = 3_500_000
    instance_count = 10
    n_steps = 1
    rewards = {'death': -1.0, 'food': 1.0, 'dec_distance': 0.1, 'inc_distance': -0.1, 'timeout': -0.05}

    evaluate = True

    # Model (Neural Network to train or evaluate)
    model = Sequential(
        [
            Dense(16, activation='relu', input_shape=(8, )),
            Dense(16, activation='relu'),
            Dense(32, activation='relu'),
            Dense(32, activation='relu')
        ]
    )

    """ 
    
    Flatten(input_shape=(3, 22, 22)),
    Dense(480, activation='relu'),
    Dense(240, activation='relu'),
    
    Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_first',
                   padding='same', input_shape=(3, 22, 22)),
            MaxPool2D((2, 2), padding='same', data_format='channels_first'),
            Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', data_format='channels_first',
                   padding='same'),
            MaxPool2D((2, 2), padding='same', data_format='channels_first'),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.15),
    """

    # Agent (object that interacts with the environment in order to learn)
    agent = ExperimentalDQN(model, 3, optimizer=Adam(lr=learning_rate), policy=EpsilonGreedy(0.5), mem_size=memory_size,
                target_update=target_network_update, gamma=gamma, batch_size=batch_size, nsteps=n_steps,
                policy_adjustment=EpsilonAdjustmentInfo(0.5, 0.0, 2_000_000, 'linear'))

    #agent = Random(3)

    # Start a new training session or evaluate an old one
    if not evaluate:
        path = create_session_info("Snake Abstract", model, learning_rate, gamma, n_steps, target_network_update,
                                   memory_size, batch_size, step_count, instance_count, rewards, seed)
        training = Training(SnakeAbstract.create, agent)
        training.train(step_count, max_subprocesses=0, checkpnt_func=chkpnt, path=path,
                       rewards=rewards)
        agent.save(os.path.join(path, "weights.dat"), True)
        training.evaluate(10_000, visualize=True, plot_func=plot_eval, rewards=rewards)
    else:
        evaluation = Evaluation(SnakeSimple.create, agent, "weights.dat", rewards)
        fails = evaluation.evaluate(max_rounds=20, max_steps=1_000, visualize=True, plot_func=plot_eval,
                                    step_delay=None)
        print(f"Failed {fails} times")


if __name__ == "__main__":
    main()
