import random
from collections import deque

import numpy as np
import pygame
from pygame.locals import *

from framework.core import Environment
from framework.environments.snake_objects import *

"""---SnakeAbstract class"""


class SnakeAbstract(Environment):
    """An abstract Reinforcement Learning representation of the snake game.

    Attributes:
        width (int): Width of the pygame window
        height (int): Height of the pygame window
        frame_count (int): Number of consecutive frames that will make up the state
        snake ((int, int)[]): Array representation of the snake with every element being a tuple (x, y)
        food ((int, int)): Tuple (x, y) that represents the food
        direction_state (enum): Current direction the snake is heading to
        last_states (frame[]): Array of the last individual frames that make up the state
        display_surf (pygame display): The surface to render to
        distance (float): The current distance from the snake head to the food
        reward_dict (dictionary): Collection of rewards that apply reward or punishment to the agent eg. for dieing
    """

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
        return cls(1000, 1000, 1, rewards)

    def __init__(self, width, height, frame_count, rewards):
        self.width = width
        self.height = height
        self.frame_count = frame_count

        multiplier_x = float(width) / 22.0
        multiplier_y = float(height) / 22.0

        self.snake = [SnakeHead((random.randint(0, 19), random.randint(0, 19)), visualize=False, size=(40, 40),
                                offset=(
                                    multiplier_x + ((multiplier_x - 40) / 2.0),
                                    multiplier_y + ((multiplier_y - 40) / 2.0)),
                                multiplier=multiplier_x)]
        self.food = Food((random.randint(0, 19), random.randint(0, 19)), visualize=False, size=(36, 45),
                         offset=(
                             multiplier_x + ((multiplier_x - 36) / 2.0), multiplier_y + ((multiplier_y - 45) / 2.0)),
                         multiplier=multiplier_x)
        self.background = Background((0, 0), visualize=False, size=(1000, 1000), offset=(0, 0), multiplier=1)

        self.direction_state = random.randint(0, 3)
        self.last_states = None
        self.display_surf = None
        self.distance = 0

        self.reward_dict = {'death': -10.0, 'food': 20.0, 'dec_distance': 3.0,
                            'inc_distance': -5.0} if rewards is None else rewards
        print(self.reward_dict)

    def reset(self):
        multiplier_x = float(self.width) / 22.0
        multiplier_y = float(self.height) / 22.0
        self.snake = [SnakeHead((random.randint(0, 19), random.randint(0, 19)),
                                visualize=False if self.display_surf is None else True, size=(40, 40),
                                offset=(
                                    multiplier_x + ((multiplier_x - 40) / 2.0),
                                    multiplier_y + ((multiplier_y - 40) / 2.0)),
                                multiplier=multiplier_x)]
        self.food.set_pos((random.randint(0, 19), random.randint(0, 19)))

        self.direction_state = random.randint(0, 3)
        self.distance = np.hypot(self.food.pos[0] - self.snake[-1].pos[0], self.food.pos[1] - self.snake[-1].pos[1])

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

        pop = True
        if self.snake[-1].pos == self.food.pos:
            self.food.set_pos((random.randint(0, 19), random.randint(0, 19)))
            multiplier_x = float(self.width) / 22.0
            multiplier_y = float(self.height) / 22.0
            # noinspection PyTypeChecker
            self.snake.insert(0, SnakeBody(pos, visualize=False if self.display_surf is None else True, size=(40, 40),
                                           offset=(multiplier_x + ((multiplier_x - 40) / 2.0),
                                                   multiplier_y + ((multiplier_y - 40) / 2.0)),
                                           multiplier=multiplier_x))
            self.distance = np.hypot(self.food.pos[0] - self.snake[-1].pos[0], self.food.pos[1] - self.snake[-1].pos[1])
            reward = self.reward_dict['food']

        if self.snake[-1].pos[0] < 0 or self.snake[-1].pos[0] > 19 or self.snake[-1].pos[1] < 0 or \
                self.snake[-1].pos[1] > 19:
            done = True
            reward = self.reward_dict['death']

        for i in range(0, len(self.snake) - 1):
            if self.snake[-1].pos == self.snake[i].pos:
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
        state = np.zeros((22, 22))

        for x in range(0, 22):
            state[x][0] = self.CELL_WALL
            state[x][21] = self.CELL_WALL

        for y in range(1, 21):
            state[0][y] = self.CELL_WALL
            state[21][y] = self.CELL_WALL

        cntr = 0
        for item in self.snake:
            state[item.pos[0] + 1][item.pos[1] + 1] = self.CELL_SNAKE_HEAD \
                if cntr == len(self.snake) - 1 else self.CELL_SNAKE_BODY
            cntr += 1

        state[self.food.pos[0] + 1][self.food.pos[1] + 1] = self.CELL_FOOD
        return state
