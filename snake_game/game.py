#!/usr/bin/env python3
import random
import logging
from typing import List, Tuple

GRID_WIDTH = 17
GRID_HEIGHT = 17

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

logging.basicConfig(level=logging.INFO)


class SnakeGame:
    def __init__(self, grid_width: int = GRID_WIDTH, grid_height: int = GRID_HEIGHT) -> None:
        self.grid_width: int = grid_width
        self.grid_height: int = grid_height
        self.reset_game()

    def reset_game(self) -> None:
        self.snake: List[Tuple[int, int]] = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction: Tuple[int, int] = random.choice([UP, DOWN, LEFT, RIGHT])
        self.place_food()
        self.score: int = 0
        self.game_over: bool = False

    def place_food(self) -> None:
        empty_cells: List[Tuple[int, int]] = [
            (x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if (x, y) not in self.snake
        ]
        self.food = random.choice(empty_cells) if empty_cells else None

    def change_direction(self, new_direction: Tuple[int, int]) -> None:
        opposite_direction: Tuple[int, int] = (-self.direction[0], -self.direction[1])
        if new_direction != opposite_direction:
            self.direction = new_direction

    def update(self) -> None:
        if self.game_over:
            return
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head: Tuple[int, int] = (head_x + dx, head_y + dy)
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

    def get_state(self) -> dict:
        return {
            "snake": self.snake.copy(),
            "food": self.food,
            "score": self.score,
            "game_over": self.game_over,
            "direction": self.direction,
        }

    def is_game_over(self) -> bool:
        return self.game_over
