import random
import time
import numpy as np
import pygame
from pygame.locals import *


class Environment:
    STATE_RIGHT = 0
    STATE_LEFT = 1
    STATE_UP = 2
    STATE_DOWN = 3
    STATE_STILL = 4

    def __init__(self, width, height, frame_count):
        self.__width = width
        self.__height = height
        self.__frame_count = frame_count

        self.__snake = [(10, 10)]
        self.__food = (random.randint(0, 19), random.randint(0, 19))
        self.__direction_state = self.STATE_STILL
        self.__last_states = []

        self.__display_surf = pygame.display.set_mode((1000, 1000))

    def reset(self):
        self.__snake = [(10, 10)]
        self.__food = (random.randint(0, 19), random.randint(0, 19))
        self.__direction_state = self.STATE_STILL

        s = self._build_current_state()
        self.__last_states = [s, s, s]

        next_state = np.array(self.__last_states)
        return next_state

    def act(self, action):
        reward = 0
        done = False

        if action == self.STATE_RIGHT:
            if not self.__direction_state == self.STATE_LEFT:
                self.__direction_state = self.STATE_RIGHT
        elif action == self.STATE_LEFT:
            if not self.__direction_state == self.STATE_RIGHT:
                self.__direction_state = self.STATE_LEFT
        elif action == self.STATE_UP:
            if not self.__direction_state == self.STATE_DOWN:
                self.__direction_state = self.STATE_UP
        elif action == self.STATE_DOWN:
            if not self.__direction_state == self.STATE_UP:
                self.__direction_state = self.STATE_DOWN

        if self.__direction_state == self.STATE_LEFT:
            self.__snake.append((self.__snake[-1][0] - 1, self.__snake[-1][1]))
        elif self.__direction_state == self.STATE_RIGHT:
            self.__snake.append((self.__snake[-1][0] + 1, self.__snake[-1][1]))
        elif self.__direction_state == self.STATE_UP:
            self.__snake.append((self.__snake[-1][0], self.__snake[-1][1] - 1))
        elif self.__direction_state == self.STATE_DOWN:
            self.__snake.append((self.__snake[-1][0], self.__snake[-1][1] + 1))

        if self.__snake[-1][0] < 0 or self.__snake[-1][0] > 19 or self.__snake[-1][1] < 0 or self.__snake[-1][1] > 19:
            done = True
            reward = -10

        for i in range(0, len(self.__snake) - 1):
            if self.__snake[-1] == self.__snake[i]:
                done = True
                reward = -10
                break

        pop = True
        if self.__snake[-1] == self.__food:
            self.__food = (random.randint(0, 19), random.randint(0, 19))
            pop = False
            reward = 10

        if not self.__direction_state == self.STATE_STILL and pop:
            self.__snake.pop(0)

        s = self._build_current_state()
        self.__last_states.pop(0)
        self.__last_states.append(s)

        next_state = np.array(self.__last_states)

        return next_state, reward, done

    def render(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        self.__display_surf.fill((0, 0, 0))

        for x in range(0, 1000, 50):
            pygame.draw.line(self.__display_surf, (100, 100, 100), (x - 1, 0), (x - 1, 1000), 2)

        for y in range(0, 1000, 50):
            pygame.draw.line(self.__display_surf, (100, 100, 100), (0, y - 1), (1000, y - 1), 2)

        for item in self.__snake:
            pygame.draw.rect(self.__display_surf, (255, 255, 255),
                             (int(item[0] * 50 + 2), int(item[1] * 50 + 2), 46, 46), 0)

        pygame.draw.circle(self.__display_surf, (200, 50, 50), (self.__food[0] * 50 + 25, self.__food[1] * 50 + 25), 25)

        pygame.display.update()

    def _build_current_state(self):
        state = np.zeros((20, 20))
        for item in self.__snake:
            if 0 <= item[0] < 20:
                if 0 <= item[1] < 20:
                    state[item[0]][item[1]] = 1

        state[self.__food[0]][self.__food[1]] = 2
        return state
