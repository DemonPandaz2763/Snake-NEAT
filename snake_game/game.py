#!/usr/bin/env python3
import random
import logging
from typing import Tuple

GRID_WIDTH = 17
GRID_HEIGHT = 17

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

logging.basicConfig(level=logging.INFO)


class SnakeGame:
    def __init__(self, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.reset_game()

    def reset_game(self):
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.place_food()
        self.score = 0
        self.game_over = False

    def place_food(self):
        empty_cells= [
            (x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if (x, y) not in self.snake
        ]
        self.food = random.choice(empty_cells) if empty_cells else None

    def change_direction(self, new_direction: Tuple[int, int]) -> None:
        opposite_direction = (-self.direction[0], -self.direction[1])
        if new_direction != opposite_direction:
            self.direction = new_direction

    def update(self):
        if self.game_over:
            return
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        if not (0 <= new_head[0] < self.grid_width and 0 <= new_head[1] < self.grid_height):
            logging.info("Game over: Snake hit the wall.")
            self.game_over = True
            return
        if new_head in self.snake:
            logging.info("Game over: Snake collided with itself.")
            self.game_over = True
            return
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            self.place_food()
        else:
            self.snake.pop()

    def get_state(self):
        return {
            "snake": self.snake.copy(),
            "food": self.food,
            "score": self.score,
            "game_over": self.game_over,
            "direction": self.direction,
        }

    def is_game_over(self):
        return self.game_over
