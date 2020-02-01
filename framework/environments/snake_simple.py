"""This file implements a simple snake environment.

    Typical usage example:
    env = SnakeSimple.create(rewards={'death': -10.0, ...})
    state = env.reset()
"""

import random
import numpy as np
import pygame
from pygame.locals import *
from framework.core import Environment


class SnakeSimple(Environment):
    """Simple Snake environment for fast learning.

    Attributes:
        width (int): Width of the pygame window
        height (int) : Height of the pygame window
        snake (array): Abstract snake representation, each list element is a two-element tuple for the x and y
                       coordinates of the snake body element.
        food (tuple): Abstract food representation, tuple with x and y coordinates
        direction_state (Enum): The current direction the snake is heading towards
        display_surf (pygame display): The surface to render on
        distance (float): Current distance to the food.
        reward_dict (dictionary): collection of rewards (eg. for death, food)
    """
    STATE_RIGHT = 0
    STATE_LEFT = 1
    STATE_UP = 2
    STATE_DOWN = 3
    STATE_STILL = 4

    @classmethod
    def create(cls, rewards=None):
        """This method creates the environment. Pass this function into the Training class.

        Args:
            rewards (dictionary): collection of rewards for different situations the agent could encounter in that
                                  environment

        Returns:
            A new object of instance SnakeSimple
        """
        return cls(1000, 1000, rewards)

    def __init__(self, width, height, rewards):
        """SnakeSimple constructor. Do not use that directly, rather use create().

        Args:
            width (int): width of the window if the game gets rendered
            height (int): height of the window if the game gets rendered
            rewards (dictionary): collection of rewards (see create())
        """
        self.width = width
        self.height = height

        self.snake = [(random.randint(0, 19), random.randint(0, 19))]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = self.STATE_STILL
        self.display_surf = None
        self.distance = 0
        self.reward_dict = {'death': -10.0, 'food': 20.0, 'dec_distance': 3.0,
                            'inc_distance': -5.0} if rewards is None else rewards
        print(self.reward_dict)

    def reset(self):
        """Resets the current environment to the default state.

        Returns:
            The new state that was created by resetting.
        """
        self.snake = [(random.randint(0, 19), random.randint(0, 19))]
        self.food = (random.randint(0, 19), random.randint(0, 19))
        self.direction_state = self.STATE_STILL
        self.distance = np.hypot(self.food[0] - self.snake[-1][0], self.food[1] - self.snake[-1][1])

        s = self._build_current_state()
        return s

    def step(self, action):
        """Moves the snake one step based on the specified action.

        Args:
            action (int): The action to do represented as a number from 0 to 3.

        Returns:
            (np.ndarray, float, boolean, None) Next state, reward, if this was a terminal step, information about the
                                               step
        """
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

        dist = np.hypot(self.food[0] - self.snake[-1][0], self.food[1] - self.snake[-1][1])

        if dist < self.distance:
            reward = self.reward_dict['dec_distance']
        else:
            reward = self.reward_dict['inc_distance']

        self.distance = dist

        pop = True
        if self.snake[-1] == self.food:
            self.food = (random.randint(0, 19), random.randint(0, 19))
            pop = False
            reward = self.reward_dict['food']

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
        return s, reward, done, None

    def render(self):
        """Renders the current state of the game using pygame."""
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
        """Builds the current state of the game into a simple representation.

        Returns:
            (np.ndarray) Simple one dimensional array with six elements, which represent the distance to each wall and
                         x and y distances to the food.
        """
        state = np.zeros(6)
        state[0] = self.snake[-1][1]
        state[1] = 19 - self.snake[-1][1]
        state[2] = self.snake[-1][0]
        state[3] = 19 - self.snake[-1][0]
        state[4] = self.snake[-1][0] - self.food[0]
        state[5] = self.snake[-1][1] - self.food[1]
        return state
