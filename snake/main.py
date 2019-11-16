# Created: 31st of October 2019
# Author: Niels Pressel

import tensorflow as tf
from DQNUtil.agent import Agent
from snake.environment import Environment


def save_score(s):
    with open("scores.txt", "w") as f:
        for item in s:
            f.write(str(item) + ", ")


EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 1e-5

BATCH_SIZE = 256
MEMORY_SIZE = 100000
LEARNING_RATE = 0.001
GAMMA = 0.999
TARGET_NET_UPDATE = 10

EPISODES = 10000
NUM_STEPS = 1000

RESTORE_FROM_CHECKPOINT = False

print("Tensorflow: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

env = Environment(20, 20, 3)
agent = Agent(LEARNING_RATE, GAMMA, 4, EPSILON_START, BATCH_SIZE, TARGET_NET_UPDATE, (3, 20, 20), EPSILON_DECAY,
              EPSILON_END, MEMORY_SIZE, "q_eval.h5", "q_target.h5")

if RESTORE_FROM_CHECKPOINT:
    agent.load_models()

scores = []

for i in range(0, EPISODES):
    print("--Starting Episode " + str(i) + "---")

    state = env.reset()

    score = 0

    for x in range(0, NUM_STEPS):
        action = agent.select_action(state, use_epsilon=not RESTORE_FROM_CHECKPOINT)

        next_state, reward, done = env.act(action)
        score += reward

        agent.push_observation(state, action, reward, next_state, int(done))

        state = next_state

        env.render()

        if not RESTORE_FROM_CHECKPOINT:
            agent.train()

        if done:
            break

    if not RESTORE_FROM_CHECKPOINT:
        print(agent.epsilon)
        scores.append(score)
        agent.save_model()
        save_score(scores)
