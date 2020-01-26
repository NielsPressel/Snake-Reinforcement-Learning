# Created: 31st of October 2019
# Author: Niels Pressel

import os
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from framework.agents.dqn import DQN
from framework.environments.snake_abstract import SnakeAbstract
from framework.environments.snake_simple import SnakeSimple
from framework.environments.snake import Snake
from framework.training import Training
from framework.policy import EpsilonGreedy, EpsilonAdjustmentInfo

from framework.evaluation import Evaluation

import gym
import time


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
                        instances):
    folder_id = time.strftime("%Y-%m-%d %H-%M-%S", time.gmtime())
    folder_path = "../weight_data/" + folder_id

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(os.path.join(folder_path, 'description.txt'), 'w') as f:
        f.write("Environment: %s \nLearning rate: %f \nGamma: %f \nnSteps: %d \nTarget network update: %d \n"
                "Memory size: %d \nBatch size: %d \nStep count: %d \nInstance count: %d \n\n"
                % (env, lr, gamma, nsteps, target_network_update, memory_size, batch_size, step_count, instances))

        model.summary(print_fn=lambda x: f.write(x + "\n"))

    return folder_path


def chkpnt(path, agent, step):
    full_path = os.path.join(path, 'checkpoints')
    if os.path.exists(full_path):
        file_list = [f for f in os.listdir(full_path)]
        for f in file_list:
            os.remove(os.path.join(full_path, f))

    agent.save(os.path.join(full_path, "checkpoint-%d.dat" % step))


def main():
    print("Tensorflow: ", tf.version.VERSION)

    tf.config.list_physical_devices('GPU')

    LEARNING_RATE = 3e-3
    GAMMA = 0.95
    TARGET_NETWORK_UPDATE = 10
    MEMORY_SIZE = 100_000
    BATCH_SIZE = 256
    STEP_COUNT = 1_000_000
    INSTANCE_COUNT = 1
    N_STEPS = 2

    evaluate = True

    model = Sequential(
        [
            Conv2D(40, kernel_size=(3, 3), strides=(1, 1), input_shape=(2, 22, 22), activation='relu', data_format='channels_first'),
            Conv2D(80, kernel_size=(3, 3), strides=(1, 1), activation='relu'),
            Flatten(),
            Dense(320, activation='relu'),
        ]
    )

    agent = DQN(model, 3, optimizer=Adam(lr=LEARNING_RATE), policy=EpsilonGreedy(0.25), mem_size=MEMORY_SIZE,
                target_update=TARGET_NETWORK_UPDATE, gamma=GAMMA, batch_size=BATCH_SIZE, nsteps=N_STEPS,
                policy_adjustment=EpsilonAdjustmentInfo(1.0, 0.1, 800_000, 'linear'))

    if not evaluate:
        path = create_session_info("Snake Abstract", model, LEARNING_RATE, GAMMA, N_STEPS, TARGET_NETWORK_UPDATE,
                                   MEMORY_SIZE, BATCH_SIZE, STEP_COUNT, INSTANCE_COUNT)
        #path = 'C:\\Users\\niels\\Documents\\Facharbeit\\weight_data\\2020-01-23 18-51-24'
        training = Training(SnakeAbstract.create, agent)
        training.train(STEP_COUNT, INSTANCE_COUNT, visualize=False, plot_func=None, max_subprocesses=0, checkpnt_func=chkpnt, path=path,
                       rewards={'death': -70.0, 'food': 40.0, 'dec_distance': 3.0, 'inc_distance': -15.0}, resume=False)
        agent.save(os.path.join(path, "weights.dat"), True)
        training.evaluate(10_000, visualize=True, plot_func=plot_eval)
    else:
        evaluation = Evaluation(SnakeAbstract.create, agent, "weights.dat")
        fails = evaluation.evaluate(max_rounds=20, max_steps=1_000, visualize=True, plot_func=plot_eval,
                                    step_delay=None)
        print("Failed %d times" % fails)


if __name__ == "__main__":
    main()