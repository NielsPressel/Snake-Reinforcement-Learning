import pygame


class GameObject:

    def __init__(self, pos, visualize=False, size=(10, 10), offset=(10, 10), multiplier=10):
        self.image = self._initialize_graphics(visualize, size)
        self.rect = pygame.Rect(int(pos[0] * multiplier + offset[0]), int(pos[1] * multiplier + offset[1]), size[0], size[1])
        self.pos = pos
        self.offset = offset
        self.multiplier = multiplier

    def _initialize_graphics(self, visualize, size):
        raise NotImplementedError()

    def load_graphics(self):
        self.image = self._initialize_graphics(True, (self.rect.width, self.rect.height))

    def set_pos(self, pos):
        self.pos = pos
        self.rect.left = int(pos[0] * self.multiplier + self.offset[0])
        self.rect.top = int(pos[1] * self.multiplier + self.offset[1])


class Food(GameObject):

    def _initialize_graphics(self, visualize, size):
        if visualize:
            image = pygame.image.load("./res/food.png").convert_alpha()
            return pygame.transform.scale(image, size)
        return None


class Background(GameObject):

    def _initialize_graphics(self, visualize, size):
        if visualize:
            image = pygame.image.load("./res/background.bmp").convert()
            return pygame.transform.scale(image, size)
        return None


class SnakeHead(GameObject):

    def _initialize_graphics(self, visualize, size):
        if visualize:
            image = pygame.image.load("./res/snake_head.png").convert_alpha()
            return pygame.transform.scale(image, size)
        return None

    def rotate(self, angle):
        if self.image is not None:
            self.image = pygame.transform.rotate(self.image, angle)


class SnakeBody(GameObject):

    def _initialize_graphics(self, visualize, size):
        if visualize:
            image = pygame.image.load("./res/snake_body.png").convert_alpha()
            return pygame.transform.scale(image, size)
        return None
