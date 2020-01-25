import random
from collections import deque

import numpy as np
import pygame
from pygame.locals import *

from framework.core import Environment


class SnakeAbstract(Environment):
    STATE_RIGHT = 0
    STATE_DOWN = 1
    STATE_LEFT = 2
    STATE_UP = 3
    STATE_STILL = 4

    ACTION_MAINTAIN = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2

    CELL_EMPTY = 0.0
    CELL_WALL = 1.0
    CELL_SNAKE_BODY = 2.0
    CELL_SNAKE_HEAD = 3.0
    CELL_FOOD = 4.0

    @classmethod
    def create(cls, rewards=None):
        return cls(1000, 1000, 2, rewards)

    def __init__(self, width, height, frame_count, rewards):
        self.width = width
        self.height = height
        self.frame_count = frame_count

        self.snake = [(random.randint(1, 18), random.randint(1, 18))]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = random.randint(0, 3)
        self.last_states = None
        self.display_surf = None
        self.distance = 0

        self.reward_dict = {'death': -10.0, 'food': 20.0, 'dec_distance': 3.0,
                            'inc_distance': -5.0} if rewards is None else rewards
        print(self.reward_dict)

    def reset(self):
        self.snake = [(random.randint(1, 18), random.randint(1, 18))]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = random.randint(0, 3)
        self.distance = np.hypot(self.food[0] - self.snake[-1][0], self.food[1] - self.snake[-1][1])

        s = self._build_current_state()
        self.last_states = deque([s] * self.frame_count)

        next_state = np.asarray(self.last_states)
        return next_state

    def step(self, action):
        done = False

        if action == self.ACTION_LEFT:
            self.direction_state -= 1
            if self.direction_state < 0:
                self.direction_state = 3
        elif action == self.ACTION_RIGHT:
            self.direction_state += 1
            if self.direction_state > 3:
                self.direction_state = 0

        if self.direction_state == self.STATE_LEFT:
            self.snake.append((self.snake[-1][0] - 1, self.snake[-1][1]))
        elif self.direction_state == self.STATE_RIGHT:
            self.snake.append((self.snake[-1][0] + 1, self.snake[-1][1]))
        elif self.direction_state == self.STATE_UP:
            self.snake.append((self.snake[-1][0], self.snake[-1][1] - 1))
        elif self.direction_state == self.STATE_DOWN:
            self.snake.append((self.snake[-1][0], self.snake[-1][1] + 1))

        dist = np.hypot(self.food[0] - self.snake[-1][0], self.food[1] - self.snake[-1][1])

        if dist < self.distance:
            reward = self.reward_dict['dec_distance'] * max(1.0, float(len(self.snake)) * 0.75)
        else:
            reward = self.reward_dict['inc_distance']

        self.distance = dist

        pop = True
        if self.snake[-1] == self.food:
            self.food = (random.randint(0, 19), random.randint(0, 19))
            pop = False
            reward = self.reward_dict['food'] * float(len(self.snake))
            self.distance = np.hypot(self.food[0] - self.snake[-1][0], self.food[1] - self.snake[-1][1])

        if not self.direction_state == self.STATE_STILL and pop:
            self.snake.pop(0)

        if self.snake[-1][0] < 0 or self.snake[-1][0] > 19 or self.snake[-1][1] < 0 or self.snake[-1][1] > 19:
            done = True
            reward = self.reward_dict['death']

        for i in range(0, len(self.snake) - 1):
            if self.snake[-1] == self.snake[i]:
                done = True
                reward = self.reward_dict['death']
                break

        s = self._build_current_state()
        if self.last_states is None:
            self.last_states = deque([s] * self.frame_count)
        else:
            self.last_states.append(s)
            self.last_states.popleft()

        next_state = np.asarray(self.last_states)
        return next_state, reward, done, None

    def render(self):
        if self.display_surf is None:
            self.display_surf = pygame.display.set_mode((self.width, self.height))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        self.display_surf.fill((0, 0, 0))

        for x in range(0, self.width, int(self.width / 20)):
            pygame.draw.line(self.display_surf, (100, 100, 100), (x - 1, 0), (x - 1, self.height), 2)

        for y in range(0, self.height, int(self.height / 20)):
            pygame.draw.line(self.display_surf, (100, 100, 100), (0, y - 1), (self.width, y - 1), 2)

        for item in self.snake:
            pygame.draw.rect(self.display_surf, (255, 255, 255),
                             (int(item[0] * int(self.width / 20) + 2),
                              int(item[1] * int(self.height / 20) + 2), 46, 46), 0)

        pygame.draw.circle(self.display_surf, (200, 50, 50),
                           (self.food[0] * int(self.width / 20) + 25, self.food[1] * int(self.height / 20) + 25), 25)

        pygame.display.update()

    def _build_current_state(self):
        state = np.zeros((22, 22))

        for x in range(0, 22):
            state[x][0] = self.CELL_WALL
            state[x][21] = self.CELL_WALL

        for y in range(1, 21):
            state[0][y] = self.CELL_WALL
            state[21][y] = self.CELL_WALL

        cntr = 0
        for item in self.snake:
            state[item[0] + 1][item[1] + 1] = self.CELL_SNAKE_HEAD if cntr == len(
                self.snake) - 1 else self.CELL_SNAKE_BODY
            cntr += 1

        state[self.food[0] + 1][self.food[1] + 1] = self.CELL_FOOD
        return state
