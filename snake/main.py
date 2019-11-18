# Created: 31st of October 2019
# Author: Niels Pressel
import tensorflow as tf
import gym
from DQNUtil.agent import Agent
from snake.environment import Environment


def save_score(s):
    with open("scores.txt", "w") as f:
        for item in s:
            f.write(str(item) + ", ")


EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 5e-4

BATCH_SIZE = 512
MEMORY_SIZE = 100000
LEARNING_RATE = 1e-4
GAMMA = 0.999
TARGET_NET_UPDATE = 10

EPISODES = 2000
NUM_STEPS = 1000

RESTORE_FROM_CHECKPOINT = False

print("Tensorflow: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


env = gym.make('CartPole-v0')
agent = Agent(LEARNING_RATE, GAMMA, env.action_space.n, EPSILON_START, BATCH_SIZE, TARGET_NET_UPDATE, 4,
              EPSILON_DECAY,
              EPSILON_END, MEMORY_SIZE)

if RESTORE_FROM_CHECKPOINT:
    agent.load_models()

scores = []

for i in range(0, EPISODES):
    print("--Starting Episode " + str(i) + "---")

    state = env.reset()

    score = 0

    for x in range(0, NUM_STEPS):
        action = agent.select_action(state, use_epsilon=not RESTORE_FROM_CHECKPOINT)

        next_state, reward, done, info = env.step(action)
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
