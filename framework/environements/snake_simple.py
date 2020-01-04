import random
import numpy as np
import pygame
from pygame.locals import *
from collections import deque
from framework.core import Environment


class SnakeSimple(Environment):
    STATE_RIGHT = 0
    STATE_LEFT = 1
    STATE_UP = 2
    STATE_DOWN = 3
    STATE_STILL = 4

    @classmethod
    def create(cls):
        return cls(1000, 1000)

    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.snake = [(10, 10)]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = self.STATE_STILL
        self.display_surf = None

    def reset(self):
        self.snake = [(10, 10)]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = self.STATE_STILL

        s = self._build_current_state()
        return s

    def step(self, action):
        reward = 0.05
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
            #reward = 1.0

        if not self.direction_state == self.STATE_STILL and pop:
            self.snake.pop(0)

        s = self._build_current_state()
        return s, reward, done, None

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
        state = np.zeros(4)
        state[0] = self.snake[0][1]
        state[1] = 20 - self.snake[0][1]
        state[2] = self.snake[0][0]
        state[3] = 20 - self.snake[0][0]
        return state
