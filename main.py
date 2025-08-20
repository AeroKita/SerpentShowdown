import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pygame


# -----------------------------
# Configuration and Data Models
# -----------------------------

GridPos = Tuple[int, int]


@dataclass
class Settings:
    grid_cells: int = 25
    move_interval_ms: int = 160  # ~6.25 moves/second for a relaxed pace
    point_spawn_interval_ms: int = 3000
    game_duration_ms: int = 2 * 60 * 1000  # 2-minute matches
    respawn_delay_ms: int = 5000
    max_player_segments_asset: int = 16
    outline_width: int = 2


class AssetLoader:
    def __init__(self, base_dir: str, tile_size: int):
        self.base_dir = base_dir
        self.tile_size = tile_size

        def load_image(name: str) -> pygame.Surface:
            path = os.path.join(self.base_dir, "assets", "images", name)
            image = pygame.image.load(path).convert_alpha()
            if image.get_width() != self.tile_size or image.get_height() != self.tile_size:
                image = pygame.transform.smoothscale(image, (self.tile_size, self.tile_size))
            return image

        # Background (grid image removed per spec; we'll draw a programmatic transparent grid)
        self.blue_bg = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "bluebg.png")).convert()

        # Characters
        self.player_head = load_image("PlayerCharacter.png")
        self.player_head_ko = load_image("PlayerCharacterKnockout.png")
        self.enemy_head = load_image("Enemy.png")
        self.enemy_segment = load_image("EnemySegment.png")
        self.point = load_image("DiamondPoint.png")
        # UI
        self.timer_ui = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "TimerUI.png")).convert_alpha()

        # Player segments 2..16
        self.player_segments: List[pygame.Surface] = []
        for i in range(2, 17):
            self.player_segments.append(load_image(f"PlayerSegment{i}.png"))

    def get_player_segment_image(self, index_from_two: int) -> pygame.Surface:
        # index_from_two: 2..N
        idx = index_from_two - 2
        if idx < 0:
            idx = 0
        if idx >= len(self.player_segments):
            # Repeat 15 after 16 per spec (index 13 for segment 15)
            return self.player_segments[13]
        return self.player_segments[idx]

    def scale_blue_background(self, width: int, height: int) -> pygame.Surface:
        return pygame.transform.smoothscale(self.blue_bg, (width, height))


class Snake:
    def __init__(
        self,
        name: str,
        start_pos: GridPos,
        start_dir: GridPos,
        head_image: pygame.Surface,
        is_player: bool,
    ):
        self.name = name
        self.start_pos = start_pos
        self.start_dir = start_dir
        self.head_image = head_image
        self.is_player = is_player

        self.reset()

    def reset(self) -> None:
        self.direction: GridPos = self.start_dir
        self.next_direction: GridPos = self.start_dir
        self.body: List[GridPos] = [self.start_pos]  # head only initially
        self.prev_body: List[GridPos] = list(self.body)
        self.pending_growth: int = 0
        self.score: int = 0
        self.eliminated: bool = False
        self.inactive_until_ms: int = 0
        # Knockout animation timing (used for player)
        self.knockout_time_ms: int = 0
        self.death_anim_end_ms: int = 0

    @property
    def head(self) -> GridPos:
        return self.body[0]

    def set_direction(self, new_dir: GridPos) -> None:
        # Prevent reversing into own neck
        if len(self.body) > 1:
            nx, ny = new_dir
            cx, cy = self.direction
            if (nx, ny) == (-cx, -cy):
                return
        self.next_direction = new_dir

    def step(self, grid_cells: int) -> None:
        if self.eliminated:
            return
        self.direction = self.next_direction
        dx, dy = self.direction
        hx, hy = self.head
        new_head = (hx + dx, hy + dy)
        self.body.insert(0, new_head)
        if self.pending_growth > 0:
            self.pending_growth -= 1
        else:
            self.body.pop()

    def collides_with_walls(self, grid_cells: int) -> bool:
        x, y = self.head
        return not (0 <= x < grid_cells and 0 <= y < grid_cells)

    def collides_with_self(self) -> bool:
        return self.head in self.body[1:]

    def collides_with_other(self, other: "Snake") -> bool:
        return self.head in other.body

    def eliminate(self, now_ms: int, respawn_delay_ms: int) -> None:
        self.eliminated = True
        self.knockout_time_ms = now_ms
        # Reserve first 500ms for disintegration/fade-in; countdown begins after
        self.death_anim_end_ms = now_ms + 500
        self.inactive_until_ms = self.death_anim_end_ms + (respawn_delay_ms)

    def try_respawn(self, now_ms: int) -> None:
        if self.eliminated and now_ms >= self.inactive_until_ms:
            self.reset()


class EnemyController:
    def __init__(self, snake: Snake):
        self.snake = snake

    @staticmethod
    def manhattan(a: GridPos, b: GridPos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def choose_direction(self, points: Sequence[GridPos], grid_cells: int, occupied: set) -> GridPos:
        if self.snake.eliminated:
            return self.snake.direction

        head = self.snake.head
        candidates = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        # Choose nearest point; if none, head to center
        target: GridPos
        if points:
            target = min(points, key=lambda p: self.manhattan(head, p))
        else:
            center = (grid_cells // 2, grid_cells // 2)
            target = center

        def is_safe_dir(d: GridPos) -> bool:
            nx = head[0] + d[0]
            ny = head[1] + d[1]
            return 0 <= nx < grid_cells and 0 <= ny < grid_cells and (nx, ny) not in occupied

        # Score directions by distance reduction while being safe
        scored: List[Tuple[int, GridPos]] = []
        for d in candidates:
            if is_safe_dir(d):
                nx = head[0] + d[0]
                ny = head[1] + d[1]
                dist = self.manhattan((nx, ny), target)
                scored.append((dist, d))

        if not scored:
            # No safe moves, keep moving; may die next step
            return self.snake.direction

        # Prefer minimal distance; tie-break randomly
        min_dist = min(s for s, _ in scored)
        best = [d for s, d in scored if s == min_dist]
        return random.choice(best)


class Game:
    def __init__(self, settings: Settings, assets: AssetLoader, screen: pygame.Surface, world_rect: pygame.Rect):
        self.settings = settings
        self.assets = assets
        self.screen = screen

        self.grid_cells = settings.grid_cells
        self.tile_size = assets.tile_size
        self.world_rect = world_rect
        self.world_size_px = (self.world_rect.width, self.world_rect.height)
        self.screen_size_px = self.screen.get_size()
        self.ui_rect = pygame.Rect(0, 0, self.screen_size_px[0], self.screen_size_px[1] - self.world_rect.height)

        # State
        self.points: List[GridPos] = []
        self.last_point_spawn_ms: int = 0
        self.last_move_ms: int = 0
        self.start_time_ms: int = 0
        self.state: str = "MENU"  # MENU, COUNTDOWN, PLAYING, GAME_OVER
        self.countdown_start_ms: int = 0
        self.result_text: str = ""
        self.high_score: int = self.load_high_score()

        # Snakes setup: near borders facing inward
        player_start = (2, self.grid_cells // 2)
        enemy1_start = (self.grid_cells - 3, 3)
        enemy2_start = (self.grid_cells - 3, self.grid_cells - 4)

        self.player = Snake("Player", player_start, (1, 0), self.assets.player_head, is_player=True)
        self.enemy1 = Snake("Enemy A", enemy1_start, (-1, 0), self.assets.enemy_head, is_player=False)
        self.enemy2 = Snake("Enemy B", enemy2_start, (-1, 0), self.assets.enemy_head, is_player=False)
        self.enemies = [self.enemy1, self.enemy2]
        self.enemy_ais = [EnemyController(self.enemy1), EnemyController(self.enemy2)]

        # Pre-scale background to window
        self.bg_blue_scaled = self.assets.scale_blue_background(*self.world_size_px)
        # Create transparent grid overlay
        self.grid_overlay = self._create_grid_overlay()

        # UI
        self.font_large = pygame.font.SysFont(None, int(self.tile_size * 1.2))
        self.font_medium = pygame.font.SysFont(None, int(self.tile_size * 0.9))
        self.font_small = pygame.font.SysFont(None, int(self.tile_size * 0.7))
        self._knockout_cache: Optional[pygame.Surface] = None

        # Prepare Timer UI scaling to fit top bar
        self._prepare_timer_ui()

    # ------------- High Score Persistence -------------
    def high_score_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "highscore.json")

    def load_high_score(self) -> int:
        try:
            with open(self.high_score_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data.get("high_score", 0))
        except Exception:
            return 0

    def save_high_score(self, score: int) -> None:
        try:
            with open(self.high_score_path(), "w", encoding="utf-8") as f:
                json.dump({"high_score": int(score)}, f)
        except Exception:
            pass

    # -------------------- Game Flow --------------------
    def start_countdown(self, now_ms: int) -> None:
        self.state = "COUNTDOWN"
        self.countdown_start_ms = now_ms
        self.reset_world()

    def start_game(self, now_ms: int) -> None:
        self.state = "PLAYING"
        self.start_time_ms = now_ms
        self.last_move_ms = now_ms
        self.last_point_spawn_ms = now_ms

    def end_game(self) -> None:
        # Player loses if they have less points than any enemy at end of 3 minutes
        player_score = self.player.score
        enemy_best = max(e.score for e in self.enemies)
        if player_score < enemy_best:
            self.result_text = "Game Over"
        else:
            self.result_text = "You Win!"
        if player_score > self.high_score:
            self.high_score = player_score
            self.save_high_score(player_score)
        self.state = "GAME_OVER"

    def reset_world(self) -> None:
        self.player.reset()
        for e in self.enemies:
            e.reset()
        self.points.clear()

    # -------------------- Update --------------------
    def update(self, now_ms: int) -> None:
        if self.state == "COUNTDOWN":
            if now_ms - self.countdown_start_ms >= 3000:
                self.start_game(now_ms)
            return

        if self.state != "PLAYING":
            return

        # Timer end
        if now_ms - self.start_time_ms >= self.settings.game_duration_ms:
            self.end_game()
            return

        # Respawns
        self.player.try_respawn(now_ms)
        for e in self.enemies:
            e.try_respawn(now_ms)

        # Spawn points periodically
        if now_ms - self.last_point_spawn_ms >= self.settings.point_spawn_interval_ms:
            self.spawn_points()
            self.last_point_spawn_ms = now_ms

        # Movement pacing (continuous interpolation used for rendering; logic steps here)
        if now_ms - self.last_move_ms >= self.settings.move_interval_ms:
            self.last_move_ms = now_ms
            self.drive_enemies()
            self.step_snakes()
            self.resolve_collisions(now_ms)

    def spawn_points(self) -> None:
        spawn_count = random.randint(1, 4)
        occupied = set(self.player.body)
        for e in self.enemies:
            occupied.update(e.body)
        for _ in range(spawn_count):
            # Try several attempts to find a free cell
            for _attempt in range(200):
                x = random.randint(0, self.grid_cells - 1)
                y = random.randint(0, self.grid_cells - 1)
                if (x, y) not in occupied and (x, y) not in self.points:
                    self.points.append((x, y))
                    break

    def drive_enemies(self) -> None:
        occupied = set(self.player.body)
        for e in self.enemies:
            occupied.update(e.body)
        for ai in self.enemy_ais:
            snake = ai.snake
            if snake.eliminated:
                continue
            # Disallow reversing by passing current direction and body
            safe_occupied = set(occupied)
            # Allow the snake to move into its own tail if it will move away this turn
            if not snake.pending_growth and len(snake.body) > 0:
                safe_occupied.discard(snake.body[-1])
            new_dir = ai.choose_direction(self.points, self.grid_cells, safe_occupied)
            snake.set_direction(new_dir)

    def step_snakes(self) -> None:
        # Record previous positions for interpolation
        self.player.prev_body = list(self.player.body)
        for e in self.enemies:
            e.prev_body = list(e.body)

        self.player.step(self.grid_cells)
        for e in self.enemies:
            e.step(self.grid_cells)

        # Eat points
        self.try_eat_points(self.player, is_player=True)
        for e in self.enemies:
            self.try_eat_points(e, is_player=False)

    def try_eat_points(self, snake: Snake, is_player: bool) -> None:
        if snake.eliminated:
            return
        if snake.head in self.points:
            self.points.remove(snake.head)
            snake.score += 1
            snake.pending_growth += 1

    def resolve_collisions(self, now_ms: int) -> None:
        snakes = [self.player] + self.enemies

        # Walls & self (head-based only)
        for s in snakes:
            if s.eliminated:
                continue
            if s.collides_with_walls(self.grid_cells) or s.collides_with_self():
                s.eliminate(now_ms, self.settings.respawn_delay_ms)

        # Pairwise collisions using head rules
        living = [s for s in snakes if not s.eliminated]
        to_eliminate = set()

        for i in range(len(living)):
            for j in range(i + 1, len(living)):
                a = living[i]
                b = living[j]
                # Head-on: both eliminated
                if a.head == b.head:
                    to_eliminate.add(a)
                    to_eliminate.add(b)
                    continue
                # Head of a into body of b (excluding b's head) => eliminate a only
                if a.head in b.body[1:]:
                    to_eliminate.add(a)
                # Head of b into body of a (excluding a's head) => eliminate b only
                if b.head in a.body[1:]:
                    to_eliminate.add(b)

        for s in to_eliminate:
            if not s.eliminated:
                s.eliminate(now_ms, self.settings.respawn_delay_ms)

    # -------------------- Render --------------------
    def draw(self) -> None:
        # Clear full screen with blue background scaled to world size for consistency
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.bg_blue_scaled, self.world_rect.topleft)
        # Transparent grid overlay within world rect
        self.screen.blit(self.grid_overlay, self.world_rect.topleft)

        if self.state == "MENU":
            self.draw_menu()
            return

        if self.state == "COUNTDOWN":
            self.draw_points()
            self.draw_snakes()
            self.draw_hud(countdown=True)
            return

        if self.state == "PLAYING":
            self.draw_points()
            self.draw_snakes()
            self.draw_hud()
            # Knockout visual: grayscale fade-in over 0.5s when player is eliminated
            if self.player.eliminated:
                self.draw_knockout_overlay()
            return

        if self.state == "GAME_OVER":
            self.draw_points()
            self.draw_snakes()
            self.draw_game_over()

    def grid_to_px(self, pos: GridPos) -> Tuple[int, int]:
        return self.world_rect.left + pos[0] * self.tile_size, self.world_rect.top + pos[1] * self.tile_size

    def _create_grid_overlay(self) -> pygame.Surface:
        overlay = pygame.Surface(self.world_size_px, pygame.SRCALPHA)
        line_color = (255, 255, 255, 40)  # faint white
        for i in range(self.grid_cells + 1):
            x = i * self.tile_size
            y = i * self.tile_size
            pygame.draw.line(overlay, line_color, (x, 0), (x, self.world_size_px[1]))
            pygame.draw.line(overlay, line_color, (0, y), (self.world_size_px[0], y))
        return overlay

    def draw_points(self) -> None:
        for p in self.points:
            self.screen.blit(self.assets.point, self.grid_to_px(p))

    def draw_snakes(self) -> None:
        # Interpolation factor between discrete moves
        if self.state == "PLAYING":
            elapsed = pygame.time.get_ticks() - self.last_move_ms
            alpha = max(0.0, min(1.0, elapsed / self.settings.move_interval_ms))
        else:
            alpha = 0.0

        # Draw enemies first, then player on top
        for e in self.enemies:
            self._draw_snake_with_outline_interpolated(e, alpha)
        self._draw_snake_with_outline_interpolated(self.player, alpha)

    def _draw_snake_with_outline_interpolated(self, snake: Snake, alpha: float) -> None:
        if not snake.body:
            return

        # If eliminated, decide visibility: enemies disappear immediately; player shows 0.5s disintegration then disappears
        if snake.eliminated:
            if not snake.is_player:
                return
            # Player: allow rendering only during disintegration window
            if pygame.time.get_ticks() >= snake.death_anim_end_ms:
                return

        # Compute per-segment pixel positions using interpolation
        positions_px: List[Tuple[float, float]] = []
        for i in range(len(snake.body)):
            cur = snake.body[i]
            prev = snake.prev_body[i] if i < len(snake.prev_body) else snake.body[i]
            gx = prev[0] + (cur[0] - prev[0]) * alpha
            gy = prev[1] + (cur[1] - prev[1]) * alpha
            px = gx * self.tile_size
            py = gy * self.tile_size
            positions_px.append((px, py))

        # Determine composite bounding box
        xs = [p[0] for p in positions_px]
        ys = [p[1] for p in positions_px]
        padding = 2  # pixels
        min_px = math.floor(min(xs)) - padding
        min_py = math.floor(min(ys)) - padding
        max_px = math.ceil(max(xs) + self.tile_size) + padding
        max_py = math.ceil(max(ys) + self.tile_size) + padding
        width = max_px - min_px
        height = max_py - min_py
        if width <= 0 or height <= 0:
            return
        composite = pygame.Surface((width, height), pygame.SRCALPHA)

        # Blit segments (tail to head) at interpolated positions
        for idx, (seg_px, seg_py) in enumerate(reversed(positions_px)):
            # Map reversed index to body index
            body_index = len(positions_px) - 1 - idx
            is_head = (body_index == 0)
            if is_head:
                if snake.is_player and snake.eliminated:
                    img = self.assets.player_head_ko
                else:
                    img = snake.head_image
            else:
                if snake.is_player:
                    segment_number = body_index + 1  # head is 1
                    segment_number = max(2, segment_number)
                    img = self.assets.get_player_segment_image(segment_number)
                else:
                    img = self.assets.enemy_segment
            # Disintegration for eliminated player: sequential tail->head over 0.5s
            if snake.is_player and snake.eliminated:
                t = pygame.time.get_ticks()
                if t < snake.death_anim_end_ms:
                    duration = 500
                    elapsed = max(0, t - snake.knockout_time_ms)
                    frac = elapsed / duration
                    # Determine how many segments have disintegrated from tail
                    seg_count = len(positions_px)
                    disintegrated = int(frac * seg_count)
                    # Skip drawing segments that have disintegrated from tail side
                    if body_index >= seg_count - disintegrated:
                        continue
            composite.blit(img, (seg_px - min_px, seg_py - min_py))

        # Outline around the union of the composite
        mask = pygame.mask.from_surface(composite)
        outline_points = mask.outline()
        if outline_points:
            pygame.draw.polygon(
                self.screen,
                (0, 0, 0),
                [(self.world_rect.left + min_px + x, self.world_rect.top + min_py + y) for (x, y) in outline_points],
                width=self.settings.outline_width,
            )

        # Blit composite
        self.screen.blit(composite, (self.world_rect.left + min_px, self.world_rect.top + min_py))

    def draw_hud(self, countdown: bool = False) -> None:
        # Scores and timer
        elapsed_ms = max(0, pygame.time.get_ticks() - self.start_time_ms)
        remaining_ms = max(0, self.settings.game_duration_ms - elapsed_ms)
        remaining_s = remaining_ms // 1000

        # Draw Timer UI background centered in top bar
        if self._timer_ui_scaled is not None:
            self.screen.blit(self._timer_ui_scaled, self._timer_ui_rect)
        # Timer text in mm:ss, pixel-like (antialias off)
        minutes = remaining_s // 60
        seconds = remaining_s % 60
        timer_str = f"{minutes}:{seconds:02d}"
        timer_font = pygame.font.SysFont(None, max(18, int(self.ui_rect.height * 0.45)))
        timer_surf = timer_font.render(timer_str, False, (255, 255, 255))
        timer_rect = timer_surf.get_rect(center=self._timer_text_center)
        self.screen.blit(timer_surf, timer_rect)

        # Scores on left side of the UI bar
        scores = f"Score: {self.player.score}   High: {self.high_score}   E1: {self.enemy1.score}   E2: {self.enemy2.score}"
        score_surf = self.font_small.render(scores, True, (255, 255, 255))
        self.screen.blit(score_surf, (self.ui_rect.left + 8, self.ui_rect.top + 8))

        if countdown:
            t = pygame.time.get_ticks() - self.countdown_start_ms
            remaining = max(0, 3000 - t)
            num = 1 + remaining // 1000
            msg = str(int(num))
            surf2 = self.font_large.render(msg, True, (255, 255, 0))
            rect = surf2.get_rect(center=(self.world_rect.centerx, self.world_rect.centery))
            self.screen.blit(surf2, rect)

        # Show respawn countdown whenever the player is knocked out
        if self.state == "PLAYING" and self.player.eliminated:
            now = pygame.time.get_ticks()
            if now >= self.player.death_anim_end_ms:
                remaining = max(0, (self.player.inactive_until_ms - now) // 1000)
                msg = f"Respawn in {remaining}s"
                overlay = self.font_large.render(msg, True, (255, 255, 255))
                rect = overlay.get_rect(center=(self.world_rect.centerx, self.world_rect.centery))
                self.screen.blit(overlay, rect)

    def draw_menu(self) -> None:
        title = self.font_large.render("Serpent Showdown", True, (255, 255, 255))
        prompt = self.font_medium.render("Click Start", True, (255, 255, 0))
        high = self.font_small.render(f"High Score: {self.high_score}", True, (200, 200, 255))

        title_rect = title.get_rect(center=(self.screen_size_px[0] // 2, int(self.screen_size_px[1] * 0.35)))
        btn_w, btn_h = int(self.tile_size * 6), int(self.tile_size * 2)
        btn_rect = pygame.Rect(0, 0, btn_w, btn_h)
        btn_rect.center = (self.screen_size_px[0] // 2, int(self.screen_size_px[1] * 0.6))

        self.screen.blit(title, title_rect)
        self.screen.blit(high, (title_rect.left, title_rect.bottom + 10))

        pygame.draw.rect(self.screen, (0, 0, 0), btn_rect, border_radius=8)
        inner = btn_rect.inflate(-4, -4)
        pygame.draw.rect(self.screen, (50, 120, 255), inner, border_radius=8)
        p_rect = prompt.get_rect(center=inner.center)
        self.screen.blit(prompt, p_rect)

        self._menu_button_rect = btn_rect  # cache for click detection

    def draw_game_over(self) -> None:
        msg = self.font_large.render(self.result_text, True, (255, 200, 50))
        info = self.font_small.render("Click to return to Menu", True, (230, 230, 230))
        rect = msg.get_rect(center=(self.screen_size_px[0] // 2, int(self.screen_size_px[1] * 0.45)))
        self.screen.blit(msg, rect)
        self.screen.blit(info, (rect.left, rect.bottom + 10))

    def draw_knockout_overlay(self) -> None:
        # Render the current frame into an offscreen surface and grayscale it, with 0.5s fade-in
        if self._knockout_cache is None or self._knockout_cache.get_size() != self.world_size_px:
            self._knockout_cache = pygame.Surface(self.world_size_px)
        # Copy current world area
        self._knockout_cache.blit(self.screen, (0, 0), area=self.world_rect)
        gray = to_grayscale(self._knockout_cache)
        # Fade factor
        if self.player.eliminated and self.player.knockout_time_ms > 0:
            t = pygame.time.get_ticks()
            duration = 500
            elapsed = max(0, t - self.player.knockout_time_ms)
            alpha = max(0.0, min(1.0, elapsed / duration))
        else:
            alpha = 1.0
        # Blend gray over current world
        gray_surface = gray.convert_alpha()
        fade = pygame.Surface(self.world_size_px, pygame.SRCALPHA)
        fade.fill((255, 255, 255, int(alpha * 255)))
        gray_surface.blit(fade, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(gray_surface, self.world_rect.topleft)
        # Countdown is drawn by draw_hud; additionally show KO text
        ko = self.font_medium.render("Knocked Out", True, (255, 255, 255))
        rect = ko.get_rect(center=(self.world_rect.centerx, int(self.world_rect.top + self.world_rect.height * 0.4)))
        self.screen.blit(ko, rect)

    def _prepare_timer_ui(self) -> None:
        # Fit TimerUI into ui_rect height with small margins; center horizontally
        margin = 6
        target_h = max(24, int((self.ui_rect.height - 2 * margin) * 0.75))  # 25% smaller
        ui = self.assets.timer_ui
        scale = target_h / ui.get_height()
        target_w = int(ui.get_width() * scale)
        self._timer_ui_scaled = pygame.transform.smoothscale(ui, (target_w, int(target_h)))
        self._timer_ui_rect = self._timer_ui_scaled.get_rect()
        self._timer_ui_rect.center = (self.ui_rect.centerx, self.ui_rect.top + self.ui_rect.height // 2)
        self._timer_text_center = self._timer_ui_rect.center

    # -------------------- Input --------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if self.state == "MENU":
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if hasattr(self, "_menu_button_rect") and self._menu_button_rect.collidepoint(event.pos):
                    self.start_countdown(pygame.time.get_ticks())
        elif self.state == "GAME_OVER":
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.state = "MENU"
        elif self.state in ("COUNTDOWN", "PLAYING"):
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.player.set_direction((1, 0))
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    self.player.set_direction((-1, 0))
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.player.set_direction((0, 1))
                elif event.key in (pygame.K_UP, pygame.K_w):
                    self.player.set_direction((0, -1))


def detect_tile_size(base_dir: str, grid_cells: int) -> int:
    # Use a comfortable default; grid image removed per spec
    return 32


def to_grayscale(surface: pygame.Surface) -> pygame.Surface:
    """Return a grayscale copy of the given surface using numpy for speed."""
    arr = pygame.surfarray.array3d(surface)
    # Luminosity method
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
    gray = np.stack((lum, lum, lum), axis=2)
    gs = pygame.surfarray.make_surface(gray)
    return gs.convert()


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pygame.init()
    pygame.display.set_caption("Serpent Showdown")

    grid_cells = 25
    # Target overall window height and ~75% world area
    target_win_h = 720
    target_world_h = int(target_win_h * 0.75)
    tile_size = max(16, target_world_h // grid_cells)
    world_h = tile_size * grid_cells
    win_h = target_win_h
    win_w = world_h  # keep square width matching world for simplicity
    ui_bar_h = win_h - world_h
    screen = pygame.display.set_mode((win_w, win_h))
    world_rect = pygame.Rect(0, win_h - world_h, world_h, world_h)

    settings = Settings(grid_cells=grid_cells)
    assets = AssetLoader(base_dir, tile_size)
    game = Game(settings, assets, screen, world_rect)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                game.handle_event(event)

        now_ms = pygame.time.get_ticks()
        game.update(now_ms)
        game.draw()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Ensure errors are visible in terminal
        print(f"Fatal error: {exc}")
        raise


