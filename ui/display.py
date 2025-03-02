import pygame as pg
import logging
import sys
import os

TEXT = (255, 248, 225)
BACKGROUND = (62, 39, 35)
SECONDARY_PANELS = (93, 64, 55)
HIGHLIGHT_MAIN = (255, 112, 67)
FOOD_RED = (255, 0, 0)

pg.init()

try:
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "fonts")
    font_path = os.path.join(assets_dir, "PressStart2P-Regular.ttf")
    font_small = pg.font.Font(font_path, 16)
    font_med = pg.font.Font(font_path, 24)
    font_big = pg.font.Font(font_path, 32)
except Exception as e:
    logging.error(f"Error loading assets: {e}")
    sys.exit(1)

def draw_text(surface, text, x, y, font, color=TEXT):
    text_obj = font.render(text, True, color)
    surface.blit(text_obj, (x, y))
    
def draw_snake_game(surface, game, game_area, window_height, cell_size):
    cell_size = game_area // game.grid_width
    draw_width = game.grid_width * cell_size
    draw_height = game.grid_height * cell_size
    offset_y = (window_height - draw_height) // 2
    
    game_surface = pg.Surface((draw_width, draw_height))
    game_surface.fill(BACKGROUND)
    pg.draw.rect(game_surface, HIGHLIGHT_MAIN, (0, 0, draw_width, draw_height), 3)
    
    snake_segments = game.get_state()['snake']
    num_segments = len(snake_segments)
    for i, (sx, sy) in enumerate(snake_segments):
        start_hue = 60
        end_hue = 300
        hue = start_hue + (end_hue - start_hue) * i / max(num_segments - 1, 1)
        color = pg.Color(0)
        color.hsva = (hue, 100, 100, 100)
        pg.draw.rect(game_surface, color, (sx * cell_size, sy * cell_size, cell_size, cell_size))

    if game.food:
        fx, fy = game.food
        pg.draw.rect(game_surface, FOOD_RED, (fx * cell_size, fy * cell_size, cell_size, cell_size))

    surface.blit(game_surface, (0, offset_y))
    
def draw_info_panel(surface, score, best_score, info, app):
    surface.fill(BACKGROUND)
    
    if app.state == "HUMAN":
        draw_text(surface, f"Score: {score}", 10, 10, font_med, TEXT)
        draw_text(surface, f"Best: {best_score}", 10, 40, font_med, TEXT)
        draw_text(surface, f"Gen: {info}", 10, 70, font_med, TEXT)
    else:
        draw_text(surface, f"Score: {score}", 10, 10, font_small, TEXT)
        draw_text(surface, f"Best: {best_score}", 10, 30, font_small, TEXT)
        draw_text(surface, f"{info}", 10, 50, font_small, TEXT)
    
def draw_main_menu(surface, window_width, window_height):
    surface.fill(BACKGROUND)
    
    title_y = int(window_height * 0.12)
    option_y = int(window_height * 0.31)
    second_option_y = int(window_height * 0.54)
    detail_y = int(window_height * 0.70)
    exit_y = int(window_height * 0.78)
    
    draw_text(surface, "Snake AI", (window_width // 2) - (font_big.size("Snake AI")[0] // 2), title_y, font_big, TEXT)
    draw_text(surface, "1: Play Snake", (window_width // 4) - (font_med.size("1: Play Snake")[0] // 2), option_y, font_med, TEXT)
    draw_text(surface, "2: Watch AI Train", (3 * window_width // 4) - (font_med.size("2: Watch AI Train")[0] // 2), option_y, font_med, TEXT)
    draw_text(surface, "3: Train AI Fast", (window_width // 2) - (font_med.size("3: Train AI Fast")[0] // 2), second_option_y, font_med, TEXT)
    draw_text(surface, "Options (1, 2, 3)", (window_width // 2) - (font_small.size("Options (1, 2, 3)")[0] // 2), detail_y, font_small, TEXT)
    draw_text(surface, "Esc: Exit", (window_width // 2) - (font_med.size("Esc: Exit")[0] // 2), exit_y, font_med, TEXT)
    pg.display.update()
    
def compute_node_depths(genome, config):
    depths = {node: 0 for node in config.genome_config.input_keys}
    changed = True
    while changed:
        changed = False
        for conn in genome.connections.values():
            if not conn.enabled:
                continue
            src, tgt = conn.key
            if src in depths:
                new_depth = depths[src] + 1
                if new_depth > depths.get(tgt, -1):
                    depths[tgt] = new_depth
                    changed = True
    return depths

def draw_neural_net(surface, genome, config, neural_net_width, neural_net_height):
    surface.fill(SECONDARY_PANELS)
    all_nodes = set(genome.nodes.keys())
    depths = compute_node_depths(genome, config)
    
    for node in all_nodes:
        if node not in depths:
            depths[node] = 0
        
    layers = {}
    for node, depth in depths.items():
        layers.setdefault(depth, []).append(node)
    
    max_depth = max(layers.keys())
    positions = {}
    margin = 20
    for depth, nodes in layers.items():
        x = margin + (depth / max_depth) * (neural_net_width - 2 * margin) if max_depth > 0 else neural_net_width // 2
        n = len(nodes)
        spacing = neural_net_height / (n + 1)
    
        for i, node in enumerate(sorted(nodes)):
            y = spacing * (i + 1)
            positions[node] = (x, y)
    
    for conn in genome.connections.values():
        if not conn.enabled:
            continue
        src, tgt = conn.key
    
        if src in positions and tgt in positions:
            src_pos = positions[src]
            tgt_pos = positions[tgt]
            color = (0, 255, 0) if conn.weight > 0 else (255, 0, 0)
            pg.draw.line(surface, color, src_pos, tgt_pos, 5)
    
    for node, pos in positions.items():
        pg.draw.circle(surface, TEXT, (int(pos[0]), int(pos[1])), 10)
    
def draw_progress_bar(surface, current, total, font, WINDOW_WIDTH, WINDOW_HEIGHT):
    gen_text = f"Gen: {current}/{total}"
    text_surface = font.render(gen_text, True, TEXT)
    text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, 50))
    surface.blit(text_surface, text_rect)
    
    bar_width = WINDOW_WIDTH - 100
    bar_height = 40
    bar_x = 50
    bar_y = WINDOW_HEIGHT // 2 - bar_height // 2
    pg.draw.rect(surface, (80, 80, 80), (bar_x, bar_y, bar_width, bar_height))
    
    progress = current / total
    filled_width = int(progress * bar_width)
    
    segments = 20
    segment_width = filled_width // segments if segments > 0 else filled_width
    for i in range(segments):
        hue = 60 + (240 * i / (segments - 1))
        color = pg.Color(0)
        color.hsva = (hue, 100, 100, 100)
        seg_x = bar_x + i * segment_width
        pg.draw.rect(surface, color, (seg_x, bar_y, segment_width, bar_height))
    
    apple_radius = bar_height // 2 - 2
    apple_x = bar_x + bar_width + apple_radius // 2
    apple_y = bar_y + bar_height // 2
    pg.draw.circle(surface, FOOD_RED, (apple_x, apple_y), apple_radius)