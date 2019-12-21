import random
import numpy as np
from collections import deque
from framework.core import Environment


class SnakeAbstract(Environment):
    STATE_RIGHT = 0
    STATE_LEFT = 1
    STATE_UP = 2
    STATE_DOWN = 3
    STATE_STILL = 4

    @classmethod
    def create(cls):
        return cls(1000, 1000, 4)

    def __init__(self, width, height, frame_count):
        self.width = width
        self.height = height
        self.frame_count = frame_count

        self.snake = [(10, 10)]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = self.STATE_STILL
        self.last_states = None

    def reset(self):
        self.snake = [(10, 10)]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = self.STATE_STILL

        s = self._build_current_state()
        self.last_states = deque([s] * self.frame_count)

        next_state = np.asarray(self.last_states).transpose()
        return next_state

    def step(self, action):
        reward = 0.1
        done = False

        if action == self.STATE_RIGHT:
            if not self.direction_state == self.STATE_LEFT:
                self.direction_state = self.STATE_RIGHT
        elif action == self.STATE_LEFT:
            if not self.direction_state == self.STATE_RIGHT:
                self.direction_state = self.STATE_LEFT
        elif action == self.STATE_UP:
            if not self.direction_state == self.STATE_DOWN:
                self.direction_state = self.STATE_UP
        elif action == self.STATE_DOWN:
            if not self.direction_state == self.STATE_UP:
                self.direction_state = self.STATE_DOWN

        if self.direction_state == self.STATE_LEFT:
            self.snake.append((self.snake[-1][0] - 1, self.snake[-1][1]))
        elif self.direction_state == self.STATE_RIGHT:
            self.snake.append((self.snake[-1][0] + 1, self.snake[-1][1]))
        elif self.direction_state == self.STATE_UP:
            self.snake.append((self.snake[-1][0], self.snake[-1][1] - 1))
        elif self.direction_state == self.STATE_DOWN:
            self.snake.append((self.snake[-1][0], self.snake[-1][1] + 1))

        if self.snake[-1][0] < 0 or self.snake[-1][0] > 19 or self.snake[-1][1] < 0 or self.snake[-1][1] > 19:
            done = True
            reward = -1.0

        for i in range(0, len(self.snake) - 1):
            if self.snake[-1] == self.snake[i]:
                done = True
                reward = -1.0
                break

        pop = True
        if self.snake[-1] == self.food:
            self.food = (random.randint(0, 19), random.randint(0, 19))
            pop = False
            reward = 1.0

        if not self.direction_state == self.STATE_STILL and pop:
            self.snake.pop(0)

        s = self._build_current_state()
        if self.last_states is None:
            self.last_states = deque([s] * self.frame_count)
        else:
            self.last_states.append(s)
            self.last_states.popleft()

        next_state = np.asarray(self.last_states).transpose()
        return next_state, reward, done, None

    def render(self):
        pass

    def _build_current_state(self):
        state = np.zeros((20, 20))
        for item in self.snake:
            if 0 <= item[0] < 20:
                if 0 <= item[1] < 20:
                    state[item[0]][item[1]] = 1

        state[self.food[0]][self.food[1]] = 2
        return state

