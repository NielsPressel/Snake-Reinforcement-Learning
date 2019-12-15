# Created: 31st of October 2019
# Author: Niels Pressel

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from framework.dqn import DQN
from framework.core import Transition


def plot_rewards(episode_rewards, episode_steps, done=False):
    plt.clf()
    plt.xlabel('Step')
    plt.ylabel('Reward')
    for ed, steps in zip(episode_rewards, episode_steps):
        plt.plot(steps, ed)
    plt.show() if done else plt.pause(0.001)  # Pause a bit so that the graph is updated


def main():
    num_steps = 5000

    print("Tensorflow: ", tf.__version__)

    create_env = lambda: gym.make('CartPole-v0').unwrapped
    env = create_env()

    model = Sequential(
        [
            Dense(16, activation="relu", input_shape=env.observation_space.shape),
            Dense(16, activation="relu"),
            Dense(16, activation="relu")
        ]
    )

    agent = DQN(model, env.action_space.n, nsteps=2)

    instances = 1
    envs = [create_env() for i in range(instances)]
    states = [env.reset() for env in envs]

    episode_reward_sequences = [[] for i in range(instances)]
    episode_step_sequences = [[] for i in range(instances)]
    episode_rewards = [0] * instances

    for step in range(num_steps):
        for i in range(instances):
            envs[i].render()
            action = agent.act(states[i])
            next_state, reward, done, _ = envs[i].step(action)
            agent.push_observation(Transition(states[i], action, reward, None if done else next_state))
            episode_rewards[i] += reward
            if done:
                episode_reward_sequences[i].append(episode_rewards[i])
                episode_step_sequences[i].append(step)
                episode_rewards[i] = 0
                plot_rewards(episode_reward_sequences, episode_step_sequences)
                states[i] = envs[i].reset()
            else:
                states[i] = next_state
        # Perform one step of the optimization
        agent.train(step)
        print(step)

    plot_rewards(episode_reward_sequences, episode_step_sequences, done=True)
    print(np.max(episode_reward_sequences))


if __name__ == "__main__":
    main()