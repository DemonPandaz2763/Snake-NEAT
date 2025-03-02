#!/usr/bin/env python3
import pygame as pg
import logging
import pickle
import neat
import sys
import os

from snake_game.game import SnakeGame
from ui.display import *
from ai.ai import *

logging.basicConfig(level=logging.INFO)

class App:
    def __init__(self):
        self.WINDOW_WIDTH = 1000
        self.WINDOW_HEIGHT = 646

        self.GAME_AREA = 646
        self.CELL_SIZE = 38

        self.INFO_AREA_WIDTH = 354
        self.INFO_AREA_HEIGHT = 196

        self.NET_AREA_WIDTH = 354
        self.NET_AREA_HEIGHT = 566
        
        pg.init()
        
        self.screen = pg.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pg.display.set_caption("Snake AI Project")
        
        self.load_assets()
        self.game = SnakeGame(grid_height=6, grid_width=6)
        
        self.state = "IDLE"
        self.generation = 0
        
        self.local_dir = os.path.dirname(__file__)
        config_path = os.path.join(self.local_dir, "config.txt")
        
        if not os.path.exists(config_path):
            logging.error(f"Config file path not found: {config_path}")
            sys.exit(1)
            
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path)
        
    def load_winner_genome(self):
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
        
    def save_winner_genome(self, winner_data):
        try:
            winner_file = os.path.join(os.path.dirname(__file__), 'winner.pkl')
            with open(winner_file, 'wb') as f:
                pickle.dump(winner_data, f)
            self.current_best_genome = winner_data
            logging.info(f"Winner genome updated for generation {winner_data['generation']}.")
        except Exception as e:
            logging.error(f"Failed to save winner genome: {e}")
        
    def load_assets(self):
        try:
            assets_dir = os.path.join(os.path.dirname(__file__), "assets", "fonts")
            font_path = os.path.join(assets_dir, "PressStart2P-Regular.ttf")
            self.font_small = pg.font.Font(font_path, 16)
            self.font_med = pg.font.Font(font_path, 24)
            self.font_big = pg.font.Font(font_path, 32)
        except Exception as e:
            logging.error(f"Error loading assets: {e}")
            sys.exit(1)
            
    def run(self):
        while self.state != "QUIT":
            if self.state == "IDLE":
                logging.info("Switching to IDLE mode")
                self.main_menu()
            elif self.state == "HUMAN":
                logging.info("Switching to HUMAN mode")
                self.human()
            elif self.state == "WATCH_TRAINING":
                logging.info("Switching to WATCH TRAINING mode")
                self.watch_training()
            elif self.state == "FAST_TRAINING":
                self.train_ai_fast()
                
            self.screen.fill(BACKGROUND)
        
        pg.quit()
        sys.exit()
        
    def main_menu(self):
        while self.state == "IDLE":
            draw_main_menu(self.screen, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    logging.info("QUIT mode enabled")
                    self.state = "QUIT"
                    
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        logging.info("Exiting to QUIT.")
                        self.state = "QUIT"
                    elif event.key == pg.K_1:
                        self.state = "HUMAN"
                    elif event.key == pg.K_2:
                        logging.info("key pressed")
                        self.state = "WATCH_TRAINING"
                    elif event.key == pg.K_3:
                        self.state = "FAST_TRAINING"
            
            pg.time.Clock().tick(144)
            
    def human(self):
        self.generation = 0
        self.best_score = 0
        input_buffer = []
        
        while self.state == "HUMAN":
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    logging.info("QUIT mode enabled")
                    self.state = "QUIT"
                    
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self.state = "IDLE"
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
            if self.game.score > self.best_score:
                self.best_score = self.game.score
                
            draw_snake_game(self.screen, self.game, self.GAME_AREA, self.WINDOW_HEIGHT, self.CELL_SIZE)
            
            info_surface = pg.Surface((self.INFO_AREA_WIDTH, self.INFO_AREA_HEIGHT))
            draw_info_panel(info_surface, self.game.score, self.best_score, self.generation, app)
            self.screen.blit(info_surface, (self.GAME_AREA, 0))
            
            pg.display.update()
            pg.time.Clock().tick(15)
            
            if self.game.is_game_over():
                self.generation += 1
                self.game.reset_game()
            
    def watch_training(self):
        checkpoint_file = os.path.join(os.path.dirname(__file__), "checkpoint.pkl")
        if os.path.exists:
            try:
                with open(checkpoint_file, "rb") as f:
                    population = pickle.load(f)
            except:
                population = neat.Population(self.config)
        else:
            population = neat.Population(self.config)
        
        while self.state == "WATCH_TRAINING":
            winner = population.run(eval_genomes_fast, 1)
            self.generation = population.generation
            
            simulate_winner_genome(winner, self, self.NET_AREA_WIDTH, self.NET_AREA_HEIGHT)
            
            if self.state != "WATCH_TRAINING":
                break
            
            try:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(population, f)
            except Exception as e:
                logging.error(f"Failed to save checkpoint: {e}")
            
            current_winner_data = self.load_winner_genome()
            saved_generation = current_winner_data.get("generation", -1)
            
            if self.generation > saved_generation:
                new_winner_data = {"generation": self.generation, "genome": winner}
                self.save_winner_genome(new_winner_data)
            
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.state = "QUIT"
                    
                elif event.type == pg.KEYDOWN:
                    if event.type == pg.K_ESCAPE:
                        self.state = "IDLE"        
    
    # def watch_training(self):
    #     best_data = self.load_winner_genome()
    #     if best_data.get("genome"):
    #         logging.info("Using best genome from saved data.")
    #         simulate_winner_genome(best_data["genome"], self, self.NET_AREA_WIDTH, self.NET_AREA_HEIGHT)
    #     else:
    #         logging.info("No saved best genome found. Running training.")
    #         checkpoint_file = os.path.join(os.path.dirname(__file__), "checkpoint.pkl")
    #         if os.path.exists(checkpoint_file):
    #             try:
    #                 with open(checkpoint_file, "rb") as f:
    #                     population = pickle.load(f)
    #             except Exception as e:
    #                 logging.error(f"Error loading checkpoint: {e}")
    #                 population = neat.Population(self.config)
    #         else:
    #             population = neat.Population(self.config)
            
    #         while self.state == "WATCH_TRAINING":
    #             winner = population.run(eval_genomes_fast, 1)
    #             self.generation = population.generation
                
    #             simulate_winner_genome(winner, self, self.NET_AREA_WIDTH, self.NET_AREA_HEIGHT)
                
    #             if self.state != "WATCH_TRAINING":
    #                 break
                
    #             try:
    #                 with open(checkpoint_file, "wb") as f:
    #                     pickle.dump(population, f)
    #             except Exception as e:
    #                 logging.error(f"Failed to save checkpoint: {e}")
                
    #             current_winner_data = self.load_winner_genome()
    #             saved_generation = current_winner_data.get("generation", -1)
    #             if self.generation > saved_generation:
    #                 new_winner_data = {"generation": self.generation, "genome": winner}
    #                 self.save_winner_genome(new_winner_data)
                
    #             for event in pg.event.get():
    #                 if event.type == pg.QUIT:
    #                     self.state = "QUIT"
    #                 elif event.type == pg.KEYDOWN:
    #                     if event.key == pg.K_ESCAPE:
    #                         self.state = "IDLE"
    
    def train_ai_fast(self):
        total_gens = 50
        population = neat.Population(self.config)
        clock = pg.time.Clock()
        
        while self.state == "FAST_TRAINING" and self.generation < total_gens:
            winner = population.run(eval_genomes_fast, 1)
        
            self.generation = population.generation
            self.screen.fill(BACKGROUND)
        
            draw_progress_bar(self.screen, self.generation, total_gens, self.font_med, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            pg.display.update()
        
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.state = "QUIT"
                    return
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    self.state = "IDLE"
                    return
        
            clock.tick(144)
        
        checkpoint_file = os.path.join(os.path.dirname(__file__), "checkpoint.pkl")
        try:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(population, f)
            logging.info(f"Checkpoint saved at generation {self.generation}.")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")        
            
        self.state = "IDLE"
    
if __name__ == "__main__":
    app = App()
    app.run()