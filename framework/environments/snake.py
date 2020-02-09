import numpy as np
import random
import pygame
from pygame.locals import *
from PIL import Image
from collections import deque

from framework.core import Environment


class Snake(Environment):
    STATE_RIGHT = 0
    STATE_LEFT = 1
    STATE_UP = 2
    STATE_DOWN = 3
    STATE_STILL = 4

    @classmethod
    def create(cls, rewards=None):
        return cls(1000, 1000, 4, 84, 84, rewards)

    def __init__(self, width, height, frame_count, input_width, input_height, rewards):
        self.width = width
        self.height = height
        self.frame_count = frame_count

        self.input_width = input_width
        self.input_height = input_height

        self.snake = [(random.randint(0, 19), random.randint(0, 19))]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = self.STATE_STILL
        self.last_states = None
        self.rewards_dict = {'death': -1.0, 'food': 1.0, 'dec_distance': 1.0, 'inc_distance': -1.0} if rewards is None\
            else rewards

        self.display_surf = None

    def reset(self):
        self.snake = [(random.randint(0, 19), random.randint(0, 19))]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = self.STATE_STILL

        s = self._build_current_state()
        self.last_states = deque([s] * self.frame_count)

        next_state = np.asarray(self.last_states)
        return next_state

    def step(self, action):
        done = False
        reward = 0.0

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

        pop = True
        if self.snake[-1] == self.food:
            self.food = (random.randint(0, 19), random.randint(0, 19))
            pop = False
            reward = self.rewards_dict['food']

        if not self.direction_state == self.STATE_STILL and pop:
            self.snake.pop(0)

        if self.snake[-1][0] < 0 or self.snake[-1][0] > 19 or self.snake[-1][1] < 0 or self.snake[-1][1] > 19:
            done = True
            reward = self.rewards_dict['death']

        for i in range(0, len(self.snake) - 1):
            if self.snake[-1] == self.snake[i]:
                done = True
                reward = self.rewards_dict['death']
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
        self.render()
        data = pygame.image.tostring(self.display_surf, 'RGB')
        image = Image.frombytes('RGB', (self.width, self.height), data)
        image = image.convert('L')
        image = image.resize((self.input_width, self.input_height))

        matrix = np.asarray(image.getdata(), dtype=np.uint8)
        matrix = (matrix - 128) / (128 - 1)

        return matrix.reshape(image.size[0], image.size[1])
