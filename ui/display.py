#!/usr/bin/env python3
import pygame as pg

CELL_SIZE = 38
GAME_AREA_WIDTH = 17 * CELL_SIZE
GAME_AREA_HEIGHT = 17 * CELL_SIZE
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 646

INFO_AREA_WIDTH = WINDOW_WIDTH - GAME_AREA_WIDTH
INFO_AREA_HEIGHT = 196
NET_AREA_WIDTH = WINDOW_WIDTH - GAME_AREA_WIDTH
NET_AREA_HEIGHT = 566

TEXT = (255, 248, 225)
BACKGROUND = (62, 39, 35)
SECONDARY_PANELS = (93, 64, 55)
HIGHLIGHT_MAIN = (255, 112, 67)
FOOD_RED = (255, 0, 0)


def draw_text(surface, text, x, y, font, color=TEXT):
    text_obj = font.render(text, True, color)
    surface.blit(text_obj, (x, y))


def draw_snake_game(surface, game):
    draw_width = GAME_AREA_WIDTH
    draw_height = GAME_AREA_HEIGHT
    offset_y = (WINDOW_HEIGHT - draw_height) // 2

    game_surface = pg.Surface((draw_width, draw_height))
    game_surface.fill(BACKGROUND)
    pg.draw.rect(game_surface, HIGHLIGHT_MAIN, (0, 0, draw_width, draw_height), 3)

    state = game.get_state()
    snake_segments = state['snake']
    num_segments = len(snake_segments)
    for i, (sx, sy) in enumerate(snake_segments):
        start_hue = 60
        end_hue = 300
        hue = start_hue + (end_hue - start_hue) * i / max(num_segments - 1, 1)
        color = pg.Color(0)
        color.hsva = (hue, 100, 100, 100)
        pg.draw.rect(game_surface, color, (sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    if state['food']:
        fx, fy = state['food']
        pg.draw.rect(game_surface, FOOD_RED, (fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    surface.blit(game_surface, (0, offset_y))


def draw_info_panel(surface, score, info, font):
    surface.fill(BACKGROUND)
    draw_text(surface, f"Score: {score:03d}", 10, 10, font, TEXT)
    if isinstance(info, (int, float)):
        draw_text(surface, f"Gen: {info}", 10, 50, font, TEXT)
    else:
        draw_text(surface, f"{info}", 10, 50, font, TEXT)


def compute_state(game) -> list:
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


def compute_node_depths(genome, config) -> dict:
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


def draw_neural_net(surface, genome, config) -> None:
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
        x = margin + (depth / max_depth) * (NET_AREA_WIDTH - 2 * margin) if max_depth > 0 else NET_AREA_WIDTH // 2
        n = len(nodes)
        spacing = NET_AREA_HEIGHT / (n + 1)
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


def draw_progress_bar(surface, current, total, font) -> None:
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
