import random
from collections import deque

import numpy as np
import pygame
from pygame.locals import *

from framework.core import Environment
from framework.environments.snake_objects import *

"""---SnakeAbstract class"""


class SnakeAbstractFramed(Environment):
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

    CELL_OOB = -1.0
    CELL_EMPTY = 0.0
    CELL_WALL = 1.0
    CELL_SNAKE_BODY = 2.0
    CELL_SNAKE_HEAD = 3.0
    CELL_FOOD = 4.0

    @classmethod
    def create(cls, rewards=None):
        return cls(1000, 1000, 3, rewards)

    def __init__(self, width, height, frame_count, rewards):
        self.width = width
        self.height = height
        self.frame_count = frame_count

        multiplier_x = float(width) / 22.0
        multiplier_y = float(height) / 22.0

        self.snake = [SnakeBody((10, 12), visualize=False, size=(40, 40),
                                offset=(multiplier_x + ((multiplier_x - 40) / 2.0),
                                        multiplier_y + ((multiplier_y - 40) / 2.0)),
                                multiplier=multiplier_x),
                      SnakeBody((10, 11), visualize=False, size=(40, 40),
                                  offset=(multiplier_x + ((multiplier_x - 40) / 2.0),
                                          multiplier_y + ((multiplier_y - 40) / 2.0)),
                                  multiplier=multiplier_x),
                      SnakeHead((10, 10), visualize=False, size=(40, 40),
                                  offset=(
                                      multiplier_x + ((multiplier_x - 40) / 2.0),
                                      multiplier_y + ((multiplier_y - 40) / 2.0)),
                                  multiplier=multiplier_x)
                      ]
        self.food = Food((random.randint(0, 19), random.randint(0, 19)), visualize=False, size=(36, 45),
                         offset=(
                             multiplier_x + ((multiplier_x - 36) / 2.0), multiplier_y + ((multiplier_y - 45) / 2.0)),
                         multiplier=multiplier_x)
        self.background = Background((0, 0), visualize=False, size=(1000, 1000), offset=(0, 0), multiplier=1)

        self.direction_state = random.randint(0, 3)
        self.last_states = None
        self.display_surf = None
        self.distance = 0
        self.steps_since_last_food = 0

        self.reward_dict = {'death': -10.0, 'food': 20.0, 'dec_distance': 3.0,
                            'inc_distance': -5.0, 'timeout': -0.1} if rewards is None else rewards
        print(self.reward_dict)

    @staticmethod
    def __log(base, x):
        return np.log(x) / np.log(base)

    def reset(self):
        multiplier_x = float(self.width) / 22.0
        multiplier_y = float(self.height) / 22.0
        self.snake = [SnakeBody((10, 12), visualize=False if self.display_surf is None else True, size=(40, 40),
                                offset=(multiplier_x + ((multiplier_x - 40) / 2.0),
                                        multiplier_y + ((multiplier_y - 40) / 2.0)),
                                multiplier=multiplier_x),
                      SnakeBody((10, 11), visualize=False if self.display_surf is None else True, size=(40, 40),
                                offset=(multiplier_x + ((multiplier_x - 40) / 2.0),
                                        multiplier_y + ((multiplier_y - 40) / 2.0)),
                                multiplier=multiplier_x),
                      SnakeHead((10, 10), visualize=False if self.display_surf is None else True, size=(40, 40),
                                offset=(
                                    multiplier_x + ((multiplier_x - 40) / 2.0),
                                    multiplier_y + ((multiplier_y - 40) / 2.0)),
                                multiplier=multiplier_x)
                      ]

        found = True
        pos = (1, 1)
        while found:
            found = False
            pos = (random.randint(0, 19), random.randint(0, 19))
            for item in self.snake:
                if item.pos == pos:
                    found = True
                    break

        self.food.set_pos(pos)
        self.direction_state = self.STATE_UP
        self.distance = np.hypot(self.food.pos[0] - self.snake[-1].pos[0], self.food.pos[1] - self.snake[-1].pos[1])
        self.steps_since_last_food = 6

        s = self._build_current_state()
        self.last_states = deque([s] * self.frame_count)

        next_state = np.asarray(self.last_states)
        return next_state

    def step(self, action, memory_adjustment=None):
        done = False
        reward = 0.0
        snake_length = len(self.snake)
        self.steps_since_last_food += 1

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

        head_pos = self.snake[-1].pos
        if head_pos[0] - 5 <= self.food.pos[0] < head_pos[0] + 5 or head_pos[1] - 5 <= self.food.pos[1] < head_pos[1] + 5:
            reward += 5 * (self.__log(snake_length, (snake_length + self.distance) / (snake_length + dist)))

        """
        if self.steps_since_last_food >= (0.7 * snake_length) + 15:
            reward -= (0.35 / snake_length)
            # if memory_adjustment:
                # memory_adjustment(-(0.5 / snake_length), (0.7 * snake_length) + 15)
        """

        self.distance = dist

        if self.snake[-1].pos == self.food.pos:
            found = True
            pos = (1, 1)
            while found:
                found = False
                pos = (random.randint(0, 19), random.randint(0, 19))
                for item in self.snake:
                    if item.pos == pos:
                        found = True
                        break

            self.food.set_pos(pos)
            multiplier_x = float(self.width) / 22.0
            multiplier_y = float(self.height) / 22.0
            # noinspection PyTypeChecker
            self.snake.insert(0, SnakeBody(pos, visualize=False if self.display_surf is None else True, size=(40, 40),
                                           offset=(multiplier_x + ((multiplier_x - 40) / 2.0),
                                                   multiplier_y + ((multiplier_y - 40) / 2.0)),
                                           multiplier=multiplier_x))
            self.distance = np.hypot(self.food.pos[0] - self.snake[-1].pos[0], self.food.pos[1] - self.snake[-1].pos[1])
            self.steps_since_last_food = 0
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

        reward = max(-1.0, min(reward, 1.0)) # Clamp reward to range [-1.0, 1.0]

        if snake_length <= 10:
            training_gap = 5
        else:
            training_gap = 0.3 * snake_length + 2

        save_experience = True
        if 0 < self.steps_since_last_food < training_gap:
            save_experience = False

        save_experience = save_experience | done

        return next_state, reward, done, save_experience

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

        head_pos = self.snake[-1].pos

        frame = state[max(0, head_pos[0] - 4):min(22, head_pos[0] + 6), max(0, head_pos[1] - 4):min(22, head_pos[1] + 6)]

        if frame.shape[1] != 10:
            arr = np.zeros((frame.shape[0], 10 - frame.shape[1]))
            arr.fill(self.CELL_OOB)
            if frame[1][0] == self.CELL_WALL or frame[2][0] == self.CELL_WALL:
                frame = np.concatenate((arr, frame), axis=1)
            elif frame[1][frame.shape[1] - 1] == self.CELL_WALL or frame[2][frame.shape[1] - 1] == self.CELL_WALL:
                frame = np.concatenate((frame, arr), axis=1)

        if frame.shape[0] != 10:
            arr = np.zeros((10 - frame.shape[0], 10))
            arr.fill(self.CELL_OOB)
            if frame[0][5] == self.CELL_WALL or frame[0][4] == self.CELL_WALL:
                frame = np.concatenate((arr, frame), axis=0)
            elif frame[frame.shape[0] - 1][5] or frame[frame.shape[0] - 1][4] == self.CELL_WALL:
                frame = np.concatenate((frame, arr), axis=0)

        assert frame.shape == (10, 10)
        return frame
