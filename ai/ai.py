import neat
import pygame as pg

from ui.display import *
from snake_game.game import SnakeGame

DIRECTION_MAP = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

def compute_state(game):
    head_x, head_y = game.snake[0]
    max_dim = max(game.grid_width, game.grid_height)
    
    directions_8 = {
        "N": (0, -1),
        "NE": (1, -1),
        "E": (1, 0),
        "SE": (1, 1),
        "S": (0, 1),
        "SW": (-1, 1),
        "W": (-1, 0),
        "NW": (-1, -1)
    }
    
    wall_info = []
    for dx, dy in directions_8.values():
        steps = 0
        x, y = head_x, head_y
        while 0 <= x < game.grid_width and 0 <= y < game.grid_height:
            steps += 1
            x += dx
            y += dy
        wall_info.append((steps - 1) / max_dim)
    
    food_info = []
    if game.food is None:
        food_x, food_y = -1, -1
    else:
        food_x, food_y = game.food
    for dx, dy in directions_8.values():
        ratio = 0
        diff_x = food_x - head_x
        diff_y = food_y - head_y
        if dx == 0 and dy != 0:
            if diff_x == 0 and (diff_y % dy == 0):
                r = diff_y // dy
                if r > 0:
                    ratio = r / max_dim
        elif dy == 0 and dx != 0:
            if diff_y == 0 and (diff_x % dx == 0):
                r = diff_x // dx
                if r > 0:
                    ratio = r / max_dim
        elif dx != 0 and dy != 0:
            if diff_x % dx == 0 and diff_y % dy == 0:
                r1 = diff_x // dx
                r2 = diff_y // dy
                if r1 == r2 and r1 > 0:
                    ratio = r1 / max_dim
        food_info.append(ratio)
    
    body_info = []
    for dx, dy in directions_8.values():
        min_ratio = None
        for segment in game.snake[1:]:
            diff_x = segment[0] - head_x
            diff_y = segment[1] - head_y
            ratio = None
            if dx == 0 and dy != 0:
                if diff_x == 0 and (diff_y % dy == 0):
                    r = diff_y // dy
                    if r > 0:
                        ratio = r
            elif dy == 0 and dx != 0:
                if diff_y == 0 and (diff_x % dx == 0):
                    r = diff_x // dx
                    if r > 0:
                        ratio = r
            elif dx != 0 and dy != 0:
                if diff_x % dx == 0 and diff_y % dy == 0:
                    r1 = diff_x // dx
                    r2 = diff_y // dy
                    if r1 == r2 and r1 > 0:
                        ratio = r1
            if ratio is not None:
                if min_ratio is None or ratio < min_ratio:
                    min_ratio = ratio
        if min_ratio is None:
            body_info.append(0)
        else:
            body_info.append(min_ratio / max_dim)
    
    current_dir = [0, 0, 0, 0]
    if game.direction == (0, -1):
        current_dir[0] = 1
    elif game.direction == (1, 0):
        current_dir[1] = 1
    elif game.direction == (0, 1):
        current_dir[2] = 1
    elif game.direction == (-1, 0):
        current_dir[3] = 1
    
    tail = game.snake[-1]
    diff_x = tail[0] - head_x
    diff_y = tail[1] - head_y
    tail_dir = [0, 0, 0, 0]
    if abs(diff_x) >= abs(diff_y):
        if diff_x > 0:
            tail_dir[1] = 1
        elif diff_x < 0:
            tail_dir[3] = 1
        else:
            if diff_y > 0:
                tail_dir[2] = 1
            elif diff_y < 0:
                tail_dir[0] = 1
    else:
        if diff_y > 0:
            tail_dir[2] = 1
        elif diff_y < 0:
            tail_dir[0] = 1
        else:
            if diff_x > 0:
                tail_dir[1] = 1
            elif diff_x < 0:
                tail_dir[3] = 1
    
    state = wall_info + food_info + body_info + current_dir + tail_dir
    return state

def manhattan_distance(pos1, pos2):
    """Compute Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def eval_genomes_fast(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = SnakeGame(10, 10)
        
        max_steps_without_food = 150
        steps_without_food = 0
        
        recent_positions = []
        memory_window = 10
        
        while not game.is_game_over() and steps_without_food < max_steps_without_food:
            snake_length = len(game.snake)
            scale_factor = max(1, 0.2 * snake_length)
            
            # Compute Manhattan distance to food before move
            if game.food is not None:
                old_distance = manhattan_distance(game.snake[0], game.food)
            else:
                old_distance = 0

            prev_score = game.score
            state = compute_state(game)
            output = net.activate(state)
            direction_index = output.index(max(output))
            new_direction = DIRECTION_MAP.get(direction_index, game.direction)
            game.change_direction(new_direction)
            game.update()
            
            # Compute Manhattan distance to food after move
            if game.food is not None:
                new_distance = manhattan_distance(game.snake[0], game.food)
            else:
                new_distance = 0
                
            if old_distance > new_distance:
                genome.fitness += scale_factor * 0.3 * (old_distance - new_distance)
            elif new_distance > old_distance:
                genome.fitness -= 0.3 * (new_distance - old_distance)
            
            # Apply a constant step penalty
            genome.fitness -= 0.5
            
            # Check for food consumption.
            if game.score > prev_score:
                genome.fitness += scale_factor * 20
                steps_without_food = 0
                recent_positions = []
            else:
                steps_without_food += 1

            # Circling / Loop Detection
            current_head = game.snake[0]
            recent_positions.append(current_head)
            if len(recent_positions) > memory_window:
                recent_positions.pop(0)
            if recent_positions.count(current_head) > 2:
                genome.fitness -= 0.5

        if game.is_game_over():
            genome.fitness -= 24

def simulate_winner_genome(winner, app, neural_net_width, neural_net_height, move_limit=150, rounds=1):
    net = neat.nn.FeedForwardNetwork.create(winner, app.config)
    clock = pg.time.Clock()
    
    for _ in range(rounds):
        app.game.reset_game()
        moves_without_food = 0

        while not app.game.is_game_over() and moves_without_food < move_limit:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    app.state = "QUIT"
                    return
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    app.state = "IDLE"
                    return

            previous_score = app.game.score

            state = compute_state(app.game)
            output = net.activate(state)
            direction_index = output.index(max(output))
            new_direction = DIRECTION_MAP.get(direction_index, app.game.direction)
            app.game.change_direction(new_direction)

            app.game.update()

            if app.game.score > previous_score:
                moves_without_food = 0
            else:
                moves_without_food += 1
                
            if app.game.score > app.best_score:
                app.best_score = app.game.score

            app.screen.fill(BACKGROUND)
            draw_snake_game(app.screen, app.game, app.GAME_AREA, app.screen.get_height(), app.CELL_SIZE)
            
            gen_value = getattr(app, "current_best_genome", {}).get("generation", app.generation)
            info_surface = pg.Surface((app.INFO_AREA_WIDTH, app.INFO_AREA_HEIGHT))
            draw_info_panel(info_surface, app.game.score, app.best_score, f"Gen: {gen_value}", app)
            app.screen.blit(info_surface, (app.GAME_AREA, 0))
            
            net_surface = pg.Surface((app.NET_AREA_WIDTH, app.NET_AREA_HEIGHT))
            draw_neural_net(net_surface, winner, app.config, neural_net_width=neural_net_width, neural_net_height=neural_net_height)
            app.screen.blit(net_surface, (app.GAME_AREA, 80))

            pg.display.update()
            clock.tick(60)