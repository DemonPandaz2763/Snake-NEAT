#!/usr/bin/env python3
import logging
from typing import Tuple

import neat
import pygame as pg
from snake_game.game import SnakeGame
from ui.display import compute_state, draw_snake_game, draw_info_panel, draw_neural_net

logging.basicConfig(level=logging.INFO)

DIRECTION_MAP = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}


def get_new_direction(net: neat.nn.FeedForwardNetwork, game: SnakeGame) -> Tuple[int, int]:
    input_vector = compute_state(game)
    output = net.activate(input_vector)
    direction_index = output.index(max(output))
    return DIRECTION_MAP.get(direction_index, game.direction)


def eval_genomes_fast(genomes, config) -> None:
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = SnakeGame()
        max_steps_without_food = 150
        steps_without_food = 0
        while not game.is_game_over() and steps_without_food < max_steps_without_food:
            prev_score = game.score
            new_direction = get_new_direction(net, game)
            game.change_direction(new_direction)
            game.update()
            if game.score > prev_score:
                genome.fitness += 10.0
                steps_without_food = 0
            else:
                steps_without_food += 1
                genome.fitness += 0.1


def simulate_winner_genome(winner, app, move_limit: int = None, rounds: int = 1) -> None:
    net = neat.nn.FeedForwardNetwork.create(winner, app.config)
    for r in range(rounds):
        app.game.reset_game()
        moves_without_food = 0
        while not app.game.is_game_over() and (move_limit is None or moves_without_food < move_limit):
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    app.state = "QUIT"
                    return
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    app.state = "IDLE"
                    return
            current_score = app.game.score
            new_direction = get_new_direction(net, app.game)
            app.game.change_direction(new_direction)
            app.game.update()
            if app.game.score > current_score:
                moves_without_food = 0
            else:
                moves_without_food += 1
            app.screen.fill((62, 39, 35))
            draw_snake_game(app.screen, app.game)
            info_surface = pg.Surface((app.screen.get_width() - (17 * 38), 196))
            draw_info_panel(info_surface, app.game.score, f"{r+1}/{rounds}", app.font_med)
            app.screen.blit(info_surface, (17 * 38, 0))
            net_surface = pg.Surface((app.screen.get_width() - (17 * 38), 566))
            draw_neural_net(net_surface, winner, app.config)
            app.screen.blit(net_surface, (17 * 38, 80))
            pg.display.update()
            pg.time.Clock().tick(144)
        pg.time.delay(50)
