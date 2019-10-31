# Created: 31st of October 2019
# Author: Niels Pressel

import random
import pygame
import time
from pygame.locals import *


class SnakeApp:
    STATE_STILL = 0
    STATE_RIGHT = 1
    STATE_LEFT = 2
    STATE_UP = 3
    STATE_DOWN = 4
    FPS = 60

    def __init__(self):
        self.__running = True
        self.__display_surf = None
        self.__size = self.__width, self.__height = 1000, 1000
        self.__snake = [(1, 1)]
        self.__food = (random.randint(2, 19), random.randint(2, 19))
        self.__direction_state = self.STATE_STILL
        self.__next_action = self.STATE_STILL
        self.__clock = pygame.time.Clock()

    def on_init(self):
        pygame.init()
        self.__display_surf = pygame.display.set_mode(self.__size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.__running = True

    def on_event(self, event):
        if event.type == QUIT:
            self.__running = False
        if event.type == KEYDOWN:
            if event.key == K_a or event.key == K_LEFT:
                if not self.__direction_state == self.STATE_RIGHT:
                    self.__next_action = self.STATE_LEFT
            elif event.key == K_d or event.key == K_RIGHT:
                if not self.__direction_state == self.STATE_LEFT:
                    self.__next_action = self.STATE_RIGHT
            elif event.key == K_w or event.key == K_UP:
                if not self.__direction_state == self.STATE_DOWN:
                    self.__next_action = self.STATE_UP
            elif event.key == K_s or event.key == K_DOWN:
                if not self.__direction_state == self.STATE_UP:
                    self.__next_action = self.STATE_DOWN

    def on_loop(self):
        if self.__snake[-1][0] and self.__snake[-1][1]:
            self.__direction_state = self.__next_action

        if self.__direction_state == self.STATE_LEFT:
            self.__snake.append((self.__snake[-1][0] - 1, self.__snake[-1][1]))
        elif self.__direction_state == self.STATE_RIGHT:
            self.__snake.append((self.__snake[-1][0] + 1, self.__snake[-1][1]))
        elif self.__direction_state == self.STATE_UP:
            self.__snake.append((self.__snake[-1][0], self.__snake[-1][1] - 1))
        elif self.__direction_state == self.STATE_DOWN:
            self.__snake.append((self.__snake[-1][0], self.__snake[-1][1] + 1))

        if self.__snake[-1][0] < 0 or self.__snake[-1][0] > 19 or self.__snake[-1][1] < 0 or self.__snake[-1][1] > 19:
            self.reset()

        for i in range(0, len(self.__snake) - 1):
            if self.__snake[-1] == self.__snake[i]:
                self.reset()
                break

        pop = True
        if self.__snake[-1] == self.__food:
            self.__food = (random.randint(0, 19), random.randint(0, 19))
            pop = False

        if not self.__direction_state == self.STATE_STILL and pop:
            self.__snake.pop(0)

    def on_render(self):
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

    def reset(self):
        self.__snake = [(1, 1)]
        self.__food = (random.randint(2, 19), random.randint(2, 19))
        self.__direction_state = self.STATE_STILL
        self.__next_action = self.STATE_STILL

    @staticmethod
    def on_cleanup():
        pygame.quit()

    def on_execute(self):
        self.on_init()
        t = time.time()

        while self.__running:
            self.__clock.tick(self.FPS)

            for event in pygame.event.get():
                self.on_event(event)

            ts = time.time()
            if ts - t >= 0.15:
                self.on_loop()
                self.on_render()
                t = ts

        self.on_cleanup()


if __name__ == "__main__":
    app = SnakeApp()
    app.on_execute()
