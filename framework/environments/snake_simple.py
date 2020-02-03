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

from framework.environments.snake_objects import Food, Background, SnakeHead, SnakeBody


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
    STATE_DOWN = 1
    STATE_LEFT = 2
    STATE_UP = 3
    STATE_STILL = 4

    ACTION_MAINTAIN = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2

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

        multiplier_x = float(width) / 22.0
        multiplier_y = float(height) / 22.0

        self.snake = [SnakeHead((random.randint(0, 19), random.randint(0, 19)), visualize=False, size=(40, 40),
                                offset=(multiplier_x + ((multiplier_x - 40) / 2.0), multiplier_y + ((multiplier_y - 40) / 2.0)),
                                multiplier=multiplier_x)]
        self.food = Food((random.randint(0, 19), random.randint(0, 19)), visualize=False, size=(36, 45),
                         offset=(multiplier_x + ((multiplier_x - 36) / 2.0), multiplier_y + ((multiplier_y - 45) / 2.0)),
                         multiplier=multiplier_x)
        self.background = Background((0, 0), visualize=False, size=(1000, 1000), offset=(0, 0), multiplier=1)

        self.direction_state = random.randint(0, 3)
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

        multiplier_x = float(self.width) / 22.0
        multiplier_y = float(self.height) / 22.0
        self.snake = [SnakeHead((random.randint(0, 19), random.randint(0, 19)), visualize=False if self.display_surf is None else True, size=(40, 40),
                                offset=(multiplier_x + ((multiplier_x - 40) / 2.0), multiplier_y + ((multiplier_y - 40) / 2.0)),
                                multiplier=multiplier_x)]
        self.food.set_pos((random.randint(0, 19), random.randint(0, 19)))
        self.direction_state = random.randint(0, 3)

        self.distance = np.hypot(self.food.pos[0] - self.snake[-1].pos[0], self.food.pos[1] - self.snake[-1].pos[1])

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

        if action == self.ACTION_LEFT:
            self.snake[-1].rotate(90.0)
            self.direction_state -= 1
            if self.direction_state < 0:
                self.direction_state = 3
        elif action == self.ACTION_RIGHT:
            self.snake[-1].rotate(-90.0)
            self.direction_state += 1
            if self.direction_state > 3:
                self.direction_state = 0

        pos = self.snake[-1].pos
        if self.direction_state == self.STATE_LEFT:
            self.snake[-1].set_pos((pos[0] - 1, pos[1]))
        elif self.direction_state == self.STATE_RIGHT:
            self.snake[-1].set_pos((pos[0] + 1, pos[1]))
        elif self.direction_state == self.STATE_UP:
            self.snake[-1].set_pos((pos[0], pos[1] - 1))
        elif self.direction_state == self.STATE_DOWN:
            self.snake[-1].set_pos((pos[0], pos[1] + 1))

        for index, item in enumerate(reversed(self.snake)):
            if index != 0:
                temp_pos = item.pos
                item.set_pos(pos)
                pos = temp_pos

        dist = np.hypot(self.food.pos[0] - self.snake[-1].pos[0], self.food.pos[1] - self.snake[-1].pos[1])

        if dist < self.distance:
            reward = self.reward_dict['dec_distance']
        else:
            reward = self.reward_dict['inc_distance']

        self.distance = dist

        if self.snake[-1].pos == self.food.pos:
            self.food.set_pos((random.randint(0, 19), random.randint(0, 19)))
            multiplier_x = float(self.width) / 22.0
            multiplier_y = float(self.height) / 22.0
            # noinspection PyTypeChecker
            self.snake.insert(0, SnakeBody(pos, visualize=False if self.display_surf is None else True, size=(40, 40),
                                           offset=(multiplier_x + ((multiplier_x - 40) / 2.0),
                                                   multiplier_y + ((multiplier_y - 40) / 2.0)),
                                            multiplier=multiplier_x))
            reward = self.reward_dict['food']

        if self.snake[-1].pos[0] < 0 or self.snake[-1].pos[0] > 19 or self.snake[-1].pos[1]< 0 or \
                self.snake[-1].pos[1] > 19:
            done = True
            reward = self.reward_dict['death']

        for i in range(0, len(self.snake) - 1):
            if self.snake[-1].pos == self.snake[i].pos:
                done = True
                reward = self.reward_dict['death']
                break

        s = self._build_current_state()
        return s, reward, done, None

    def render(self):
        """Renders the current state of the game using pygame."""
        if self.display_surf is None:
            self.display_surf = pygame.display.set_mode((self.width, self.height))
            self.background.load_graphics()
            self.food.load_graphics()

            for item in self.snake:
                item.load_graphics()

            if self.direction_state == self.STATE_LEFT:
                self.snake[-1].rotate(-90.0)
            elif self.direction_state == self.STATE_UP:
                self.snake[-1].rotate(180.0)
            elif self.direction_state == self.STATE_RIGHT:
                self.snake[-1].rotate(90.0)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()

        self.display_surf.blit(self.background.image, (0, 0))

        for item in self.snake:
            self.display_surf.blit(item.image, item.rect)

        self.display_surf.blit(self.food.image, self.food.rect)

        pygame.display.update()

    def _build_current_state(self):
        """Builds the current state of the game into a simple representation.

        Returns:
            (np.ndarray) Simple one dimensional array with six elements, which represent the distance to each wall and
                         x and y distances to the food.
        """
        state = np.zeros(8)
        state[0] = self.direction_state == self.STATE_UP
        state[1] = self.direction_state == self.STATE_RIGHT
        state[2] = self.direction_state == self.STATE_DOWN
        state[3] = self.direction_state == self.STATE_LEFT
        state[4] = self.snake[-1].pos[0] > self.food.pos[0]
        state[5] = self.snake[-1].pos[0] < self.food.pos[0]
        state[6] = self.snake[-1].pos[1] > self.food.pos[1]
        state[7] = self.snake[-1].pos[1] < self.food.pos[1]
        return state
