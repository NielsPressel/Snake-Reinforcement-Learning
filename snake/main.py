# Created: 31st of October 2019
# Author: Niels Pressel

import numpy as np
import tensorflow as tf
from DQNUtil.experience import Experience
from DQNUtil.strategy import EpsilonGreedyStrategy
from DQNUtil.agent import Agent
from DQNUtil.replay_memory import ReplayMemory
from snake.environment import Environment

EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 0.001

BATCH_SIZE = 32
MEMORY_SIZE = 100000
LEARNING_RATE = 0.001
GAMMA = 0.999
TARGET_NET_UPDATE = 10

EPISODES = 10000
NUM_STEPS = 1000

tf.keras.backend.clear_session()

print("Tensorflow: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

env = Environment(20, 20, 3)
strategy = EpsilonGreedyStrategy(EPSILON_START, EPSILON_END, EPSILON_DECAY)
agent = Agent(strategy, 20, 20, 3, LEARNING_RATE, 4)
memory = ReplayMemory(MEMORY_SIZE)

for i in range(0, EPISODES):
    state = env.reset()

    for x in range(0, NUM_STEPS):
        action = agent.select_action(np.reshape(state, [1, 20, 20, 3]))

        next_state, reward, done = env.act(action)
        memory.push(Experience(state, action, reward, next_state))

        state = next_state

        env.render()

        if memory.is_sample_available(BATCH_SIZE):
            agent.train(memory, BATCH_SIZE, GAMMA)

        if done:
            break

    if EPISODES % TARGET_NET_UPDATE == 0:
        agent.align_target_network()

    agent.save_model("test1", i)