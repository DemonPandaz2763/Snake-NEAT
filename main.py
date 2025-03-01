#!/usr/bin/env python3
import os
import sys
import logging
import pickle

import pygame as pg
import neat

from snake_game.game import SnakeGame
from ui.display import draw_text, draw_snake_game, draw_info_panel, draw_progress_bar
from ai.trainer import eval_genomes_fast, simulate_winner_genome

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 646
CELL_SIZE = 38
GAME_AREA_WIDTH = 17 * CELL_SIZE
INFO_AREA_WIDTH = WINDOW_WIDTH - GAME_AREA_WIDTH
INFO_AREA_HEIGHT = 196
NET_AREA_WIDTH = WINDOW_WIDTH - GAME_AREA_WIDTH
NET_AREA_HEIGHT = 566

BACKGROUND = (62, 39, 35)
TEXT = (255, 248, 225)

logging.basicConfig(level=logging.INFO)

class App:
    def __init__(self) -> None:
        pg.init()
        self.screen = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pg.display.set_caption("Snake AI Project")
        self.load_assets()
        self.game = SnakeGame()
        self.state = "IDLE"
        self.generation = 0

        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config.txt')
        if not os.path.exists(config_path):
            logging.error(f"Config file not found: {config_path}")
            sys.exit(1)

        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        self.current_best_genome = self.load_winner_genome()

    def load_assets(self) -> None:
        try:
            assets_dir = os.path.join(os.path.dirname(__file__), "assets", "fonts")
            font_path = os.path.join(assets_dir, "PressStart2P-Regular.ttf")
            self.font_small = pg.font.Font(font_path, 16)
            self.font_med = pg.font.Font(font_path, 24)
            self.font_big = pg.font.Font(font_path, 32)
        except Exception as e:
            logging.error(f"Error loading assets: {e}")
            sys.exit(1)

    def load_winner_genome(self) -> dict:
        try:
            winner_file = os.path.join(os.path.dirname(__file__), 'winner.pkl')
            with open(winner_file, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'generation' in data:
                return data
            return {'generation': -1, 'genome': data}
        except Exception as e:
            logging.error(f"Failed to load winner genome: {e}")
            return {'generation': -1, 'genome': None}

    def save_winner_genome(self, winner_data: dict) -> None:
        try:
            winner_file = os.path.join(os.path.dirname(__file__), 'winner.pkl')
            with open(winner_file, 'wb') as f:
                pickle.dump(winner_data, f)
            self.current_best_genome = winner_data
            logging.info(f"Winner genome updated for generation {winner_data['generation']}.")
        except Exception as e:
            logging.error(f"Failed to save winner genome: {e}")

    def run(self) -> None:
        while self.state != "QUIT":
            if self.state == "IDLE":
                self.main_menu()
            elif self.state == "HUMAN":
                self.human_loop()
            elif self.state == "WATCH_TRAINING":
                self.watch_training()
            elif self.state == "FAST_TRAIN":
                self.train_ai_fast()
            elif self.state == "WATCH_TRAINED":
                self.watch_trained_ai()
            self.screen.fill(BACKGROUND)
            pg.display.update()
        pg.quit()
        sys.exit()

    def main_menu(self) -> None:
        running = True
        clock = pg.time.Clock()
        while running and self.state == "IDLE":
            self.screen.fill(BACKGROUND)
            draw_text(self.screen, "Snake AI Project", (WINDOW_WIDTH // 2) - (self.font_big.size("Snake AI Project")[0] // 2), 75, self.font_big, TEXT)
            draw_text(self.screen, "1: Play Snake", (WINDOW_WIDTH // 4) - (self.font_med.size("1: Play Snake")[0] // 2), 200, self.font_med, TEXT)
            draw_text(self.screen, "2: Watch AI Train", (3 * WINDOW_WIDTH // 4) - (self.font_med.size("2: Watch AI Train")[0] // 2), 200, self.font_med, TEXT)
            draw_text(self.screen, "3: Train AI Fast", (WINDOW_WIDTH // 4) - (self.font_med.size("3: Train AI Fast")[0] // 2), 350, self.font_med, TEXT)
            draw_text(self.screen, "4: Watch Best AI", (3 * WINDOW_WIDTH // 4) - (self.font_med.size("4: Watch Best AI")[0] // 2), 350, self.font_med, TEXT)
            draw_text(self.screen, "Options (1, 2, 3, 4)", (WINDOW_WIDTH // 2) - (self.font_small.size("Options (1, 2, 3, 4)")[0] // 2), 450, self.font_small, TEXT)
            draw_text(self.screen, "Esc: Exit", (WINDOW_WIDTH // 2) - (self.font_med.size("Esc: Exit")[0] // 2), 500, self.font_med, TEXT)
            pg.display.update()

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    logging.info("Exiting to QUIT.")
                    self.state = "QUIT"
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        logging.info("Exiting to QUIT.")
                        self.state = "QUIT"
                        running = False
                    elif event.key == pg.K_1:
                        logging.info("Switching to HUMAN mode.")
                        self.state = "HUMAN"
                        running = False
                    elif event.key == pg.K_2:
                        logging.info("Switching to WATCH_TRAINING mode.")
                        self.state = "WATCH_TRAINING"
                        running = False
                    elif event.key == pg.K_3:
                        logging.info("Switching to FAST_TRAIN mode.")
                        self.state = "FAST_TRAIN"
                        running = False
                    elif event.key == pg.K_4:
                        logging.info("Switching to WATCH_TRAINED mode.")
                        self.state = "WATCH_TRAINED"
                        running = False

            clock.tick(60)

    def human_loop(self) -> None:
        self.game.reset_game()
        input_buffer = []
        running = True
        clock = pg.time.Clock()
        while running and self.state == "HUMAN":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.state = "QUIT"
                    running = False
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self.state = "IDLE"
                        running = False
                    elif event.key == pg.K_UP:
                        input_buffer.append((0, -1))
                    elif event.key == pg.K_DOWN:
                        input_buffer.append((0, 1))
                    elif event.key == pg.K_LEFT:
                        input_buffer.append((-1, 0))
                    elif event.key == pg.K_RIGHT:
                        input_buffer.append((1, 0))
            if input_buffer:
                self.game.change_direction(input_buffer.pop(0))
            self.game.update()
            draw_snake_game(self.screen, self.game)
            info_surface = pg.Surface((INFO_AREA_WIDTH, INFO_AREA_HEIGHT))
            draw_info_panel(info_surface, self.game.score, self.generation, self.font_med)
            self.screen.blit(info_surface, (GAME_AREA_WIDTH, 0))
            pg.display.update()
            clock.tick(10)
            if self.game.is_game_over():
                self.generation += 1
                self.game.reset_game()

    def watch_training(self) -> None:
        checkpoint_file = os.path.join(os.path.dirname(__file__), 'checkpoint.pkl')
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    population = pickle.load(f)
                logging.info(f"Loaded checkpoint from {checkpoint_file}")
            except Exception as e:
                logging.error(f"Failed to load checkpoint: {e}")
                population = neat.Population(self.config)
        else:
            population = neat.Population(self.config)
        while self.state == "WATCH_TRAINING":
            winner = population.run(eval_genomes_fast, 1)
            self.generation = population.generation
            simulate_winner_genome(winner, self, move_limit=150, rounds=1)
            if self.state != "WATCH_TRAINING":
                break
            try:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(population, f)
            except Exception as e:
                logging.error(f"Failed to save checkpoint: {e}")
            current_winner_data = self.load_winner_genome()
            saved_generation = current_winner_data.get('generation', -1)
            if self.generation > saved_generation:
                new_winner_data = {'generation': self.generation, 'genome': winner}
                self.save_winner_genome(new_winner_data)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    logging.info("Exiting to QUIT from WATCH_TRAINING.")
                    self.state = "QUIT"
                    return
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    logging.info("Exiting to IDLE from WATCH_TRAINING.")
                    self.state = "IDLE"
                    return

    def train_ai_fast(self) -> None:
        total_gens = 1000
        population = neat.Population(self.config)
        clock = pg.time.Clock()
        while self.state == "FAST_TRAIN" and self.generation < total_gens:
            winner = population.run(eval_genomes_fast, 1)
            self.generation = population.generation
            self.screen.fill(BACKGROUND)
            draw_progress_bar(self.screen, self.generation, total_gens, self.font_med)
            pg.display.update()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.state = "QUIT"
                    return
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    self.state = "IDLE"
                    return
            clock.tick(144)
        self.save_winner_genome({'generation': self.generation, 'genome': winner})
        self.state = "IDLE"

    def watch_trained_ai(self) -> None:
        winner_data = self.load_winner_genome()
        if winner_data['genome'] is None:
            logging.error("No trained winner genome available.")
            self.state = "IDLE"
            return
        simulate_winner_genome(winner_data['genome'], self, move_limit=150, rounds=20)


if __name__ == "__main__":
    app = App()
    app.run()
