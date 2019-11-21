# Created: 31st of October 2019
# Author: Niels Pressel

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from DQNUtil.agent import Agent
from DQNUtil.session_manager import  SessionManager
from snake.environment import Environment


EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 1e-4

BATCH_SIZE = 512
MEMORY_SIZE = 100000
LEARNING_RATE = 0.0005
GAMMA = 0.9
TARGET_NET_UPDATE = 10

EPISODES = 2000
NUM_STEPS = 1000

RESTORE_FROM_CHECKPOINT = False

print("Tensorflow: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Starting training with learning rate ", LEARNING_RATE)

# env = gym.make('CartPole-v0')

env = Environment(1000, 1000, 4, 84, 84)
sess = SessionManager()
agent = Agent(LEARNING_RATE, GAMMA, 4, EPSILON_START, BATCH_SIZE, TARGET_NET_UPDATE, (84, 84, 4),
              EPSILON_DECAY,
              EPSILON_END, MEMORY_SIZE)

sess.write_session_info(LEARNING_RATE, EPSILON_DECAY, GAMMA, agent.q_eval, TARGET_NET_UPDATE, MEMORY_SIZE,  BATCH_SIZE,
                        EPISODES, NUM_STEPS)

if RESTORE_FROM_CHECKPOINT:
    agent.load_models("")

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
        agent.save_model(sess.get_folder_path())
        sess.write_score(scores)
