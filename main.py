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
    move_interval_ms: int = 128  # 20% faster than 160ms
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
        self.enemy_head_ko = load_image("enemyKnockout.png")
        self.enemy_segment = load_image("EnemySegment.png")
        self.point = load_image("DiamondPoint.png")
        self.point_white = load_image("DiamondPointWhite.png")
        # UI
        self.timer_ui = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "TimerUI.png")).convert_alpha()
        self.fight_ui_dark = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "FightUIDark.png")).convert_alpha()
        self.fight_ui_light = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "FightUILight.png")).convert_alpha()
        self.ko_effect = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "KOEffect.png")).convert_alpha()
        self.times_up = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "TimesUp.png")).convert_alpha()
        self.logo = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "SSLogo.png")).convert_alpha()
        self.victory_logo = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "VictoryLogo.png")).convert_alpha()
        self.arrow_keys = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "ArrowKeys.png")).convert_alpha()
        self.controls_text = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "ControlsText.png")).convert_alpha()
        self.movement_text = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "MovementText.png")).convert_alpha()
        self.pause_text = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "PauseText.png")).convert_alpha()
        self.esc_key = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "EscKey.png")).convert_alpha()
        self.high_score_icon = pygame.image.load(os.path.join(self.base_dir, "assets", "images", "HighScore.png")).convert_alpha()
        # Sounds (optional)
        try:
            self.sfx_knockout = pygame.mixer.Sound(os.path.join(self.base_dir, "assets", "sounds", "KnockoutSound.mp3"))
            self.sfx_revive = pygame.mixer.Sound(os.path.join(self.base_dir, "assets", "sounds", "ReviveSound.mp3"))
            self.sfx_point = pygame.mixer.Sound(os.path.join(self.base_dir, "assets", "sounds", "PointSound.mp3"))
            self.sfx_knockout.set_volume(0.4)
            self.sfx_revive.set_volume(0.32)
            self.sfx_point.set_volume(0.32)
        except Exception:
            self.sfx_knockout = None
            self.sfx_revive = None
            self.sfx_point = None

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
        # KO effect placement/animation
        self.ko_effect_start_ms: int = 0
        self.ko_effect_anchor_px: Tuple[int, int] = (0, 0)
        self.ko_effect_dir_px: Tuple[int, int] = (0, 0)

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
        # Start KO effect
        self.ko_effect_start_ms = now_ms

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
        # UI area: fixed 450x120 centered at top
        ui_height = self.screen_size_px[1] - self.world_rect.height
        ui_width = 450
        self.ui_rect = pygame.Rect(0, 0, ui_width, ui_height)
        self.ui_rect.centerx = self.screen_size_px[0] // 2
        self.ui_rect.top = 0

        # State
        self.points: List[GridPos] = []
        self.point_teasers: List[Tuple[GridPos, int]] = []  # (cell, start_ms)
        self.last_point_spawn_ms: int = 0
        self.last_move_ms: int = 0
        self.start_time_ms: int = 0
        self.state: str = "MENU"  # MENU, COUNTDOWN, PREFIGHT, PLAYING, PAUSED, PAUSE_COUNTDOWN, GAME_OVER
        self.countdown_start_ms: int = 0
        self.prefight_start_ms: int = 0
        self.pause_start_ms: int = 0
        self.pause_countdown_start_ms: int = 0
        self.result_text: str = ""
        self.high_score: int = self.load_high_score()
        self._paused_elapsed_ms: int = 0
        self._times_up_start_ms: int = 0
        self.postgame_phase: Optional[str] = None  # None, "TIMES_UP", "TALLY", "RESULT"
        self._tally_index: int = 0
        self._tally_count: int = 0
        self._tally_last_tick_ms: int = 0
        self._tally_counts: List[int] = [0, 0, 0]
        # Match duration for this round (default 2 minutes)
        self.current_match_duration_ms: int = self.settings.game_duration_ms
        # Whether to skip tally counters at end (winner by default conditions)
        self.skip_tally: bool = False
        # Menu Start flashing state
        self.menu_start_triggered_ms: int = 0
        self.menu_flash_start_button: bool = False

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
        self.bg_blue_full = self.assets.scale_blue_background(*self.screen_size_px)
        # Create transparent grid overlay
        self.grid_overlay = self._create_grid_overlay()

        # UI
        self.font_large = pygame.font.SysFont(None, int(self.tile_size * 1.2))
        self.font_medium = pygame.font.SysFont(None, int(self.tile_size * 0.9))
        self.font_small = pygame.font.SysFont(None, int(self.tile_size * 0.7))
        self.font_counter = pygame.font.SysFont(None, int(self.tile_size * 1.8))
        # 3x larger for countdown and end-game banners
        self.font_huge = pygame.font.SysFont(None, int(self.tile_size * 1.2 * 3))
        # 75% larger than medium for Start button label
        self.font_button = pygame.font.SysFont(None, int(self.tile_size * 0.9 * 1.75))
        self._knockout_cache: Optional[pygame.Surface] = None

        # Prepare Timer UI scaling to fit top bar
        self._prepare_timer_ui()
        # Menu grid overlay toggle
        self.menu_grid_overlay = self._create_menu_grid_overlay()
        self.show_menu_grid: bool = False

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
    def start_countdown(self, now_ms: int, duration_ms: Optional[int] = None) -> None:
        self.state = "COUNTDOWN"
        self.countdown_start_ms = now_ms
        # Set per-round match duration
        self.current_match_duration_ms = (
            duration_ms if duration_ms is not None else self.settings.game_duration_ms
        )
        self.reset_world()
        self.menu_start_triggered_ms = 0
        if hasattr(self, '_menu_pending_duration_ms'):
            delattr(self, '_menu_pending_duration_ms')
        # Spawn 3 point teasers during countdown so they "spawn in" before play starts
        occupied = set(self.player.body)
        for e in self.enemies:
            occupied.update(e.body)
        spawned = 0
        attempts = 0
        while spawned < 3 and attempts < 500:
            attempts += 1
            x = random.randint(0, self.grid_cells - 1)
            y = random.randint(0, self.grid_cells - 1)
            cell = (x, y)
            if cell in occupied:
                continue
            if cell in self.points or any(t[0] == cell for t in self.point_teasers):
                continue
            self.point_teasers.append((cell, now_ms))
            spawned += 1

    def start_game(self, now_ms: int) -> None:
        self.state = "PLAYING"
        self.start_time_ms = now_ms
        self.last_move_ms = now_ms
        self.last_point_spawn_ms = now_ms

    def start_prefight(self, now_ms: int) -> None:
        self.state = "PREFIGHT"
        self.prefight_start_ms = now_ms

    def end_game(self) -> None:
        # Determine winner at time-up
        living = [s for s in [self.player] + self.enemies if not s.eliminated]
        player_score = self.player.score
        enemy_best = max((e.score for e in self.enemies), default=0)
        # If only one entity is alive on the field, they win by default (skip tally)
        if len(living) == 1:
            self.result_text = "You Win!" if living[0] is self.player else "Game Over"
            self.skip_tally = True
        else:
            # Fall back to score comparison
            self.result_text = "Game Over" if player_score < enemy_best else "You Win!"
            self.skip_tally = False
        if player_score > self.high_score:
            self.high_score = player_score
            self.save_high_score(player_score)
        self._times_up_start_ms = pygame.time.get_ticks()
        self.postgame_phase = "TIMES_UP"
        self._tally_index = 0
        self._tally_count = 0
        self._tally_last_tick_ms = 0
        self._tally_counts = [0, 0, 0]
        self.state = "GAME_OVER"

    def _pause_game(self) -> None:
        if self.state not in ("PLAYING", "COUNTDOWN"):
            return
        self.state = "PAUSED"
        self.pause_start_ms = pygame.time.get_ticks()

    def _resume_with_countdown(self) -> None:
        if self.state != "PAUSED":
            return
        self.state = "PAUSE_COUNTDOWN"
        self.pause_countdown_start_ms = pygame.time.get_ticks()

    def _finalize_resume_from_pause(self, now_ms: int) -> None:
        # Compute total paused time and shift all timers so gameplay resumes seamlessly
        paused_duration = now_ms - self.pause_start_ms + (now_ms - self.pause_countdown_start_ms)
        # Shift core timers
        self.start_time_ms += paused_duration
        self.last_move_ms += paused_duration
        self.last_point_spawn_ms += paused_duration
        # Shift player/enemy knockout timers if applicable
        for s in [self.player] + self.enemies:
            if s.eliminated:
                s.knockout_time_ms += paused_duration
                s.death_anim_end_ms += paused_duration
                s.inactive_until_ms += paused_duration
        self.state = "PLAYING"

    def reset_world(self) -> None:
        self.player.reset()
        for e in self.enemies:
            e.reset()
        self.points.clear()
        # Reset gameplay timers so a new match shows full 2:00 until start
        self.start_time_ms = 0
        self.last_move_ms = 0
        self.last_point_spawn_ms = 0
        # Reset postgame visuals
        self._times_up_start_ms = 0
        self.postgame_phase = None
        self._tally_index = 0
        self._tally_count = 0
        self._tally_last_tick_ms = 0
        self._tally_counts = [0, 0, 0]
        self.skip_tally = False

    # -------------------- Update --------------------
    def update(self, now_ms: int) -> None:
        if self.state == "COUNTDOWN":
            if now_ms - self.countdown_start_ms >= 3000:
                self.start_prefight(now_ms)
            return

        if self.state == "PREFIGHT":
            # 1s total prefight animation then start game
            if now_ms - self.prefight_start_ms >= 1000:
                self.start_game(now_ms)
            return

        # Handle menu start flashing delay before leaving MENU
        if self.state == "MENU" and self.menu_start_triggered_ms:
            if now_ms - self.menu_start_triggered_ms >= 2000:
                # Decide duration: dev test if specified
                duration = getattr(self, '_menu_pending_duration_ms', None)
                if hasattr(self, '_menu_pending_duration_ms'):
                    delattr(self, '_menu_pending_duration_ms')
                self.start_countdown(now_ms, duration_ms=duration if duration is not None else self.settings.game_duration_ms)
            return

        if self.state == "PAUSED":
            # Frozen
            return

        if self.state == "PAUSE_COUNTDOWN":
            if now_ms - self.pause_countdown_start_ms >= 3000:
                self._finalize_resume_from_pause(now_ms)
            return

        if self.state != "PLAYING":
            return

        # Timer end
        if now_ms - self.start_time_ms >= self.current_match_duration_ms:
            self.end_game()
            return

        # Respawns
        player_was_elim = self.player.eliminated
        enemy_was_elim = [e.eliminated for e in self.enemies]
        self.player.try_respawn(now_ms)
        for e in self.enemies:
            e.try_respawn(now_ms)
        # Play revive sfx on transitions
        if player_was_elim and not self.player.eliminated and getattr(self.assets, 'sfx_revive', None):
            try:
                self.assets.sfx_revive.play()
            except Exception:
                pass
        for was, e in zip(enemy_was_elim, self.enemies):
            if was and not e.eliminated and getattr(self.assets, 'sfx_revive', None):
                try:
                    self.assets.sfx_revive.play()
                except Exception:
                    pass

        # Spawn points periodically
        if now_ms - self.last_point_spawn_ms >= self.settings.point_spawn_interval_ms:
            self.spawn_points()
            self.last_point_spawn_ms = now_ms

        # Promote teasers to real points after 0.6s
        if self.point_teasers:
            new_teasers: List[Tuple[GridPos, int]] = []
            for cell, start in self.point_teasers:
                if now_ms - start >= 600:
                    if cell not in self.points:
                        self.points.append(cell)
                else:
                    new_teasers.append((cell, start))
            self.point_teasers = new_teasers

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
                cell = (x, y)
                if cell not in occupied and cell not in self.points and all(t[0] != cell for t in self.point_teasers):
                    # Start a teaser 0.4s before point appears
                    self.point_teasers.append((cell, pygame.time.get_ticks()))
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
            # Play point SFX for player only
            if is_player and getattr(self.assets, 'sfx_point', None):
                try:
                    self.assets.sfx_point.play()
                except Exception:
                    pass

    def resolve_collisions(self, now_ms: int) -> None:
        snakes = [self.player] + self.enemies

        # Walls & self (head-based only)
        for s in snakes:
            if s.eliminated:
                continue
            if s.collides_with_walls(self.grid_cells):
                # Stop at boundary by reverting to previous in-bounds position
                s.body = list(s.prev_body)
                self._set_ko_effect_for_snake(s)
                if getattr(self.assets, 'sfx_knockout', None):
                    try:
                        self.assets.sfx_knockout.play()
                    except Exception:
                        pass
                s.eliminate(now_ms, self.settings.respawn_delay_ms)
            elif s.collides_with_self():
                self._set_ko_effect_for_snake(s)
                if getattr(self.assets, 'sfx_knockout', None):
                    try:
                        self.assets.sfx_knockout.play()
                    except Exception:
                        pass
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
                self._set_ko_effect_for_snake(s)
                if getattr(self.assets, 'sfx_knockout', None):
                    try:
                        self.assets.sfx_knockout.play()
                    except Exception:
                        pass
                s.eliminate(now_ms, self.settings.respawn_delay_ms)

    # -------------------- Render --------------------
    def draw(self) -> None:
        # Clear and draw background layers based on state
        self.screen.fill((0, 0, 0))
        if self.state == "MENU":
            # On title, use full-screen background and no top UI bar area
            self.screen.blit(self.bg_blue_full, (0, 0))
        else:
            # In matches and other states, show world-only background with UI bar
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

        if self.state == "PREFIGHT":
            self.draw_points()
            self.draw_snakes()
            self.draw_hud()
            self.draw_prefight()
            return

        if self.state == "PLAYING":
            self.draw_points()
            self.draw_snakes()
            self.draw_hud()
            # Knockout visual: grayscale fade-in over 0.5s when player is eliminated
            if self.player.eliminated:
                self.draw_knockout_overlay()
            # KO effects for any eliminated snakes
            self.draw_ko_effects()
            return

        if self.state == "PAUSED":
            self.draw_points()
            self.draw_snakes()
            self.draw_hud()
            self.draw_ko_effects()
            self.draw_pause_menu()
            return

        if self.state == "PAUSE_COUNTDOWN":
            self.draw_points()
            self.draw_snakes()
            self.draw_hud()
            self.draw_ko_effects()
            self.draw_pause_countdown()
            return

        if self.state == "GAME_OVER":
            self.draw_points()
            self.draw_snakes()
            self.draw_ko_effects()
            if self.postgame_phase == "TIMES_UP":
                self.draw_times_up()
                if pygame.time.get_ticks() - self._times_up_start_ms >= 1500:
                    # Begin tallying scores unless skipping tally (winner by default)
                    if self.skip_tally:
                        self.postgame_phase = "RESULT"
                    else:
                        self.postgame_phase = "TALLY"
                        self._tally_index = 0  # 0: enemy1, 1: enemy2, 2: player
                        self._tally_count = 0
                        self._tally_last_tick_ms = pygame.time.get_ticks()
                        self._tally_counts = [0, 0, 0]
            elif self.postgame_phase == "TALLY":
                self.draw_tally_counters()
            else:
                # RESULT or fallback: keep counters on screen and show final message
                self.draw_tally_counters()
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

    def _create_menu_grid_overlay(self) -> pygame.Surface:
        overlay = pygame.Surface(self.screen_size_px, pygame.SRCALPHA)
        line_color = (255, 255, 255, 40)
        cols = (self.screen_size_px[0] // self.tile_size) + 1
        rows = (self.screen_size_px[1] // self.tile_size) + 1
        for i in range(cols + 1):
            x = i * self.tile_size
            pygame.draw.line(overlay, line_color, (x, 0), (x, self.screen_size_px[1]))
        for j in range(rows + 1):
            y = j * self.tile_size
            pygame.draw.line(overlay, line_color, (0, y), (self.screen_size_px[0], y))
        return overlay

    def draw_points(self) -> None:
        # Draw teasers (white) with 0.6s lead-in: scale pulse and fade
        now = pygame.time.get_ticks()
        for cell, start in self.point_teasers:
            t = now - start
            if t < 600:
                alpha = int(255 * (t / 600.0))
                # Pulse scale 0.9->1.1->1.0 across 0.6s
                phase = t / 600.0
                if phase < 0.5:
                    scale = 0.9 + 0.2 * (phase / 0.5)
                else:
                    scale = 1.1 - 0.1 * ((phase - 0.5) / 0.5)
                img = self.assets.point_white.copy()
                img = pygame.transform.smoothscale(img, (int(self.tile_size * scale), int(self.tile_size * scale)))
                img.set_alpha(alpha)
                px, py = self.grid_to_px(cell)
                rect = img.get_rect(center=(px + self.tile_size // 2, py + self.tile_size // 2))
                self.screen.blit(img, rect)
            else:
                # After 0.6s, keep teaser visible during COUNTDOWN/PREFIGHT until promotion
                if self.state in ("COUNTDOWN", "PREFIGHT"):
                    img = self.assets.point_white.copy()
                    img = pygame.transform.smoothscale(img, (self.tile_size, self.tile_size))
                    img.set_alpha(255)
                    px, py = self.grid_to_px(cell)
                    rect = img.get_rect(center=(px + self.tile_size // 2, py + self.tile_size // 2))
                    self.screen.blit(img, rect)
        # Draw actual points
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

        # If eliminated, decide visibility: show 0.5s disintegration then disappear for both player and enemies
        if snake.eliminated:
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
                if snake.eliminated:
                    img = self.assets.player_head_ko if snake.is_player else self.assets.enemy_head_ko
                else:
                    img = snake.head_image
            else:
                if snake.is_player:
                    segment_number = body_index + 1  # head is 1
                    segment_number = max(2, segment_number)
                    img = self.assets.get_player_segment_image(segment_number)
                else:
                    img = self.assets.enemy_segment
            # Disintegration for eliminated snake: sequential tail->head over 0.5s
            if snake.eliminated:
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
        if self.state in ("MENU", "COUNTDOWN", "PREFIGHT"):
            # Always display full duration until actual gameplay starts
            remaining_ms = self.current_match_duration_ms
        else:
            # Freeze during pause and pause-countdown; otherwise tick normally
            if self.state in ("PAUSED", "PAUSE_COUNTDOWN"):
                now_time = self.pause_start_ms
            else:
                now_time = pygame.time.get_ticks()
            elapsed_ms = max(0, now_time - self.start_time_ms)
            remaining_ms = max(0, self.current_match_duration_ms - elapsed_ms)
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

        # (Removed per request) scores in top UI bar

        if countdown:
            t = pygame.time.get_ticks() - self.countdown_start_ms
            remaining = max(0, 3000 - t)
            num = 1 + remaining // 1000
            msg = str(int(num))
            # 40% larger than font_huge
            enlarged_font = pygame.font.SysFont(None, int(self.font_huge.get_height() * 1.4))
            surf2 = enlarged_font.render(msg, True, (255, 255, 255))
            rect = surf2.get_rect(center=(self.world_rect.centerx, self.world_rect.centery))
            # Draw blue box with black border behind number
            pad = int(self.tile_size * 0.6)
            box_outer = rect.inflate(pad * 2, pad)
            pygame.draw.rect(self.screen, (0, 0, 0), box_outer, border_radius=10)
            box_inner = box_outer.inflate(-4, -4)
            pygame.draw.rect(self.screen, (50, 120, 255), box_inner, border_radius=8)
            self.screen.blit(surf2, surf2.get_rect(center=box_inner.center))

        # Show respawn countdown whenever the player is knocked out
        if self.state == "PLAYING" and self.player.eliminated:
            now = pygame.time.get_ticks()
            if now >= self.player.death_anim_end_ms:
                remaining = max(0, (self.player.inactive_until_ms - now) // 1000)
                msg = f"Respawn in {remaining}s"
                respawn_font = pygame.font.SysFont(None, int(self.font_large.get_height() * 1.5))
                overlay = respawn_font.render(msg, True, (255, 255, 255))
                rect = overlay.get_rect(center=(self.world_rect.centerx, self.world_rect.centery))
                self.screen.blit(overlay, rect)

    def draw_pause_menu(self) -> None:
        # Dim overlay across whole screen
        dim = pygame.Surface(self.screen_size_px, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        self.screen.blit(dim, (0, 0))

        # Enlarge and move down Paused title
        paused_font = pygame.font.SysFont(None, self.font_large.get_height() + self.tile_size)
        title = paused_font.render("Paused", True, (255, 255, 255))
        title_rect = title.get_rect(center=(self.screen_size_px[0] // 2, int(self.screen_size_px[1] * 0.35) + self.tile_size))
        self.screen.blit(title, title_rect)

        # Enlarge buttons by 1 tile in both dimensions
        btn_w, btn_h = int(self.tile_size * 6) + self.tile_size, int(self.tile_size * 2) + self.tile_size
        gap = int(self.tile_size * 0.8)
        # Continue button
        cont_rect = pygame.Rect(0, 0, btn_w, btn_h)
        cont_rect.center = (self.screen_size_px[0] // 2, int(self.screen_size_px[1] * 0.5) + self.tile_size)
        # Main menu button
        menu_rect = pygame.Rect(0, 0, btn_w, btn_h)
        menu_rect.center = (self.screen_size_px[0] // 2, cont_rect.bottom + gap + btn_h // 2)

        # Draw buttons
        for rect, text in ((cont_rect, "Continue"), (menu_rect, "Main Menu")):
            pygame.draw.rect(self.screen, (0, 0, 0), rect, border_radius=8)
            inner = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, (50, 120, 255), inner, border_radius=8)
            label = self.font_medium.render(text, True, (255, 255, 255))
            self.screen.blit(label, label.get_rect(center=inner.center))

        self._pause_continue_rect = cont_rect
        self._pause_menu_rect = menu_rect

    def draw_pause_countdown(self) -> None:
        dim = pygame.Surface(self.screen_size_px, pygame.SRCALPHA)
        dim.fill((0, 0, 0, 150))
        self.screen.blit(dim, (0, 0))
        t = pygame.time.get_ticks() - self.pause_countdown_start_ms
        remaining = max(0, 3000 - t)
        num = 1 + remaining // 1000
        msg = str(int(num))
        big_font = pygame.font.SysFont(None, self.font_large.get_height() + 3 * self.tile_size)
        surf = big_font.render(msg, True, (255, 255, 255))
        rect = surf.get_rect(center=(self.world_rect.centerx, self.world_rect.centery))
        self.screen.blit(surf, rect)

    def draw_menu(self) -> None:
        # Logo (move up by 20%, scale +10%)
        max_w = int(self.screen_size_px[0] * 0.88)
        scale = min(1.0, max_w / max(1, self.assets.logo.get_width())) * 1.1
        logo = pygame.transform.smoothscale(
            self.assets.logo,
            (int(self.assets.logo.get_width() * scale), int(self.assets.logo.get_height() * scale)),
        )
        prompt = self.font_button.render("Start", True, (255, 255, 0))
        # High score icon + value
        hs_h = int(self.tile_size * 2.0)
        hs_scale = hs_h / max(1, self.assets.high_score_icon.get_height())
        hs_w = int(self.assets.high_score_icon.get_width() * hs_scale)
        hs_img = pygame.transform.smoothscale(self.assets.high_score_icon, (hs_w, hs_h))

        title_rect = logo.get_rect(center=(self.screen_size_px[0] // 2, int(self.screen_size_px[1] * 0.30 * 0.8)))
        btn_w, btn_h = int(self.tile_size * 6 * 1.2), int(self.tile_size * 2 * 1.2)
        btn_rect = pygame.Rect(0, 0, btn_w + self.tile_size, btn_h + self.tile_size)
        btn_rect.center = (self.screen_size_px[0] // 2, int(self.screen_size_px[1] * 0.6 * 0.8))
        # Move Start down by 1 tile (net effect: up one tile compared to previous +2)
        btn_rect.centery += 1 * self.tile_size

        self.screen.blit(logo, title_rect)
        # Draw High Score icon and value to the right with 1 tile gap
        hs_rect = hs_img.get_rect(midleft=(title_rect.left, title_rect.bottom + hs_h // 2))
        self.screen.blit(hs_img, hs_rect)
        # High score number in Start button yellow, inside blue box with black border; font height increased by 1 tile
        hs_font = pygame.font.SysFont(None, int(self.tile_size * 1.7))
        score_txt = hs_font.render(str(self.high_score), True, (255, 255, 0))
        score_rect = score_txt.get_rect(midleft=(hs_rect.right + self.tile_size, hs_rect.centery))
        # Draw black border box then blue inner box
        pad = 8
        box_outer = score_rect.inflate(pad * 2, pad)
        pygame.draw.rect(self.screen, (0, 0, 0), box_outer, border_radius=6)
        box_inner = box_outer.inflate(-4, -4)
        pygame.draw.rect(self.screen, (50, 120, 255), box_inner, border_radius=6)
        self.screen.blit(score_txt, score_txt.get_rect(center=box_inner.center))

        pygame.draw.rect(self.screen, (0, 0, 0), btn_rect, border_radius=8)
        inner = btn_rect.inflate(-4, -4)
        # Determine flashing state if triggered
        flash_active = False
        if self.menu_start_triggered_ms:
            elapsed = pygame.time.get_ticks() - self.menu_start_triggered_ms
            # Flash for 1s: 3 flashes within 1 second
            if elapsed < 1000:
                phase = elapsed / 1000.0
                flash_points = [0.17, 0.50, 0.83]
                flash_active = any(abs(phase - fp) < 0.09 for fp in flash_points)
        # Draw button, flashing whole button when active
        if self.menu_start_triggered_ms and flash_active:
            pygame.draw.rect(self.screen, (230, 230, 230), inner, border_radius=8)
        else:
            pygame.draw.rect(self.screen, (50, 120, 255), inner, border_radius=8)
        # Start label constant
        start_label = prompt
        p_rect = start_label.get_rect(center=inner.center)
        self.screen.blit(start_label, p_rect)

        self._menu_button_rect = btn_rect  # cache for click detection

        # Dev Test Arena button (small, bottom-left)
        dev_w, dev_h = int(self.tile_size * 4), int(self.tile_size * 1.2)
        dev_btn = pygame.Rect(16, self.screen_size_px[1] - dev_h - 16, dev_w, dev_h)
        pygame.draw.rect(self.screen, (0, 0, 0), dev_btn, border_radius=6)
        inner2 = dev_btn.inflate(-4, -4)
        pygame.draw.rect(self.screen, (120, 50, 255), inner2, border_radius=6)
        dev_label = self.font_small.render("Dev Test Arena", True, (255, 255, 255))
        self.screen.blit(dev_label, dev_label.get_rect(center=inner2.center))
        self._menu_dev_button_rect = dev_btn

        # Controls blurb under Start using provided images
        # "Controls:" banner moved up by 2 tiles relative to previous
        ctrl_h = int(self.tile_size * 1.2)
        scale = ctrl_h / max(1, self.assets.controls_text.get_height())
        ctrl_w = int(self.assets.controls_text.get_width() * scale)
        ctrl_img = pygame.transform.smoothscale(self.assets.controls_text, (ctrl_w, ctrl_h))
        controls_rect = ctrl_img.get_rect(center=(self.screen_size_px[0] // 2, btn_rect.bottom + int(self.tile_size * 1.8)))
        self.screen.blit(ctrl_img, controls_rect)

        # Movement line with icon left-aligned (text 125%, icon 3.7 tiles), moved down by 2 tiles
        mv_h = int(self.tile_size * 1.25)
        mv_scale = mv_h / max(1, self.assets.movement_text.get_height())
        mv_w = int(self.assets.movement_text.get_width() * mv_scale)
        mv_img = pygame.transform.smoothscale(self.assets.movement_text, (mv_w, mv_h))
        # Place Movement text: 6 tiles from left, 7 tiles up from bottom (down by 2 tiles from 9)
        move_rect = mv_img.get_rect(bottomleft=(int(self.tile_size * 6), self.screen_size_px[1] - int(self.tile_size * 7)))
        self.screen.blit(mv_img, move_rect)
        # Arrow keys to the right of Movement by 1 tile, follows movement vertical
        arrow_h = int(self.tile_size * 3.7)
        a_scale = arrow_h / max(1, self.assets.arrow_keys.get_height())
        arrow_w = int(self.assets.arrow_keys.get_width() * a_scale)
        arrow_img = pygame.transform.smoothscale(self.assets.arrow_keys, (arrow_w, arrow_h))
        arrow_rect = arrow_img.get_rect(midleft=(move_rect.right + 1 * self.tile_size, move_rect.centery))
        self.screen.blit(arrow_img, arrow_rect)

        # Pause line left-aligned (text 125%) 3 tiles up from bottom (down by 2 tiles from 5)
        pause_h = int(self.tile_size * 1.25)
        p_scale = pause_h / max(1, self.assets.pause_text.get_height())
        pause_w = int(self.assets.pause_text.get_width() * p_scale)
        pause_img = pygame.transform.smoothscale(self.assets.pause_text, (pause_w, pause_h))
        pause_rect = pause_img.get_rect(bottomleft=(int(self.tile_size * 6), self.screen_size_px[1] - int(self.tile_size * 3)))
        self.screen.blit(pause_img, pause_rect)
        # ESC icon to the right of Pause by 2 tiles at 2.5 tiles tall
        esc_h = int(self.tile_size * 2.5)
        e_scale = esc_h / max(1, self.assets.esc_key.get_height())
        esc_w = int(self.assets.esc_key.get_width() * e_scale)
        esc_img = pygame.transform.smoothscale(self.assets.esc_key, (esc_w, esc_h))
        esc_rect = esc_img.get_rect(midleft=(pause_rect.right + 2 * self.tile_size, pause_rect.centery))
        self.screen.blit(esc_img, esc_rect)

        # Grid toggle removed per request

    def draw_game_over(self) -> None:
        if self.result_text == "You Win!":
            # Replace win text with VictoryLogo image
            img = self.assets.victory_logo
            max_w = int(self.world_rect.width * 0.7)
            scale = min(1.0, max_w / max(1, img.get_width()))
            target = pygame.transform.smoothscale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
            rect = target.get_rect(center=(self.screen_size_px[0] // 2, int(self.screen_size_px[1] * 0.45)))
            self.screen.blit(target, rect)
        else:
            msg = self.font_huge.render(self.result_text, True, (255, 200, 50))
            rect = msg.get_rect(center=(self.screen_size_px[0] // 2, int(self.screen_size_px[1] * 0.45)))
            # Draw blue box with black border behind Game Over text
            pad = 16
            box_outer = rect.inflate(pad * 2, pad)
            pygame.draw.rect(self.screen, (0, 0, 0), box_outer, border_radius=10)
            box_inner = box_outer.inflate(-4, -4)
            pygame.draw.rect(self.screen, (50, 120, 255), box_inner, border_radius=8)
            self.screen.blit(msg, msg.get_rect(center=box_inner.center))

    def draw_tally_counters(self) -> None:
        # Tick the counter faster (75% faster than prior 250ms -> ~143ms)
        now = pygame.time.get_ticks()
        order = [self.enemy1, self.enemy2, self.player]
        # Only advance counts if we are still tallying and index is valid
        if self.postgame_phase == "TALLY" and self._tally_index < len(order):
            current = order[self._tally_index]
            target = len(current.body)
            if self._tally_last_tick_ms == 0:
                self._tally_last_tick_ms = now
            if now - self._tally_last_tick_ms >= 143:
                self._tally_last_tick_ms = now
                if self._tally_count < target:
                    self._tally_count += 1
                    self._tally_counts[self._tally_index] = self._tally_count
                else:
                    # Move to next entity or finish
                    self._tally_index += 1
                    if self._tally_index >= len(order):
                        # Done tallying; lock into result phase
                        self.postgame_phase = "RESULT"
                    else:
                        self._tally_count = 0

        # Draw counters as black boxes with large white text above each head; persist after finish
        for idx, snake in enumerate(order):
            if self.postgame_phase == "RESULT":
                count_val = self._tally_counts[idx]
            else:
                count_val = self._tally_counts[idx] if (idx < self._tally_index) else (self._tally_count if idx == self._tally_index else 0)
            head_px = self.grid_to_px(snake.head)
            cx = head_px[0] + self.tile_size // 2
            cy = head_px[1] - int(self.tile_size * 0.8)
            text = self.font_counter.render(str(count_val), False, (255, 255, 255))
            pad = 8
            box = text.get_rect()
            box.inflate_ip(pad * 2, pad)
            box.center = (cx, cy)
            pygame.draw.rect(self.screen, (0, 0, 0), box, border_radius=6)
            self.screen.blit(text, text.get_rect(center=box.center))

    def draw_times_up(self) -> None:
        if self._times_up_start_ms <= 0:
            return
        t = pygame.time.get_ticks() - self._times_up_start_ms
        if t > 1500:
            return
        img = self.assets.times_up
        # Make it large: ~70% of world width
        max_w = int(self.world_rect.width * 0.7)
        scale = max_w / img.get_width()
        target = pygame.transform.smoothscale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
        rect = target.get_rect(center=(self.world_rect.centerx, self.world_rect.centery))
        self.screen.blit(target, rect)

    def draw_knockout_overlay(self) -> None:
        # Render the current frame into an offscreen surface and grayscale it, with 0.5s fade-in
        if self._knockout_cache is None or self._knockout_cache.get_size() != self.world_size_px:
            self._knockout_cache = pygame.Surface(self.world_size_px)
        # Copy current world area
        self._knockout_cache.blit(self.screen, (0, 0), area=self.world_rect)
        gray = to_grayscale(self._knockout_cache)
        # Fade factor: fade in over first 0.5s; fade out over last 2s before respawn
        alpha = 1.0
        if self.player.eliminated and self.player.knockout_time_ms > 0:
            t = pygame.time.get_ticks()
            fade_in_end = self.player.death_anim_end_ms
            fade_out_start = self.player.inactive_until_ms - 2000
            if t < fade_in_end:
                # Fade in 0..1 over 0.5s
                alpha = max(0.0, min(1.0, (t - self.player.knockout_time_ms) / 500.0))
            elif t >= fade_out_start:
                # Fade out 1..0 over last 2s
                alpha = max(0.0, min(1.0, (self.player.inactive_until_ms - t) / 2000.0))
        # Blend gray over current world according to alpha factor
        gray_surface = gray.convert_alpha()
        fade = pygame.Surface(self.world_size_px, pygame.SRCALPHA)
        fade.fill((255, 255, 255, int(alpha * 255)))
        gray_surface.blit(fade, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(gray_surface, self.world_rect.topleft)
        # Countdown is drawn by draw_hud; additionally show KO text
        ko_font = pygame.font.SysFont(None, int(self.font_medium.get_height() * 1.5))
        ko = ko_font.render("Knocked Out", True, (255, 255, 255))
        rect = ko.get_rect(center=(self.world_rect.centerx, int(self.world_rect.top + self.world_rect.height * 0.4)))
        self.screen.blit(ko, rect)

    def _prepare_timer_ui(self) -> None:
        # Fit TimerUI into ui_rect height (120) with small margins; center horizontally
        margin = 6
        target_h = max(24, int((self.ui_rect.height - 2 * margin) * 0.75))  # keep 25% smaller styling
        ui = self.assets.timer_ui
        scale = target_h / ui.get_height()
        target_w = int(ui.get_width() * scale)
        self._timer_ui_scaled = pygame.transform.smoothscale(ui, (target_w, int(target_h)))
        self._timer_ui_rect = self._timer_ui_scaled.get_rect()
        self._timer_ui_rect.center = (self.ui_rect.centerx, self.ui_rect.top + self.ui_rect.height // 2)
        self._timer_text_center = self._timer_ui_rect.center

    def draw_prefight(self) -> None:
        # 1s total: two flashes in the first 0.5s, then disappear
        t_ms = pygame.time.get_ticks() - self.prefight_start_ms
        center = (self.world_rect.centerx, self.world_rect.centery)
        # Scale fight UI to about 60% of world width
        max_w = int(self.world_rect.width * 0.6)
        # Choose current frame (light primary, dark flashes twice at ~125ms and ~375ms for ~100ms each)
        show_dark = (125 <= t_ms < 225) or (375 <= t_ms < 475)
        img = self.assets.fight_ui_dark if show_dark else self.assets.fight_ui_light
        scale = max_w / img.get_width()
        target_size = (int(img.get_width() * scale), int(img.get_height() * scale))
        frame = pygame.transform.smoothscale(img, target_size)
        rect = frame.get_rect(center=center)
        self.screen.blit(frame, rect)

    def _set_ko_effect_for_snake(self, snake: Snake) -> None:
        # Anchor near head position at tile center, slightly forward-right relative to direction
        head_x, head_y = snake.head
        base_px = self.world_rect.left + head_x * self.tile_size + self.tile_size // 2
        base_py = self.world_rect.top + head_y * self.tile_size + self.tile_size // 2
        # Offset diagonally from head depending on direction
        dx, dy = snake.direction
        off_x = (dx - dy) * (self.tile_size // 2)
        off_y = (dx + dy) * (-self.tile_size // 2)
        snake.ko_effect_anchor_px = (base_px + off_x, base_py + off_y)
        snake.ko_effect_start_ms = pygame.time.get_ticks()

    def draw_ko_effects(self) -> None:
        # KO effect lasts 1s from ko_effect_start_ms and bounces in/out
        effect_img = self.assets.ko_effect
        # Base size ~2.0x tile (100% larger than head size)
        base_size = int(self.tile_size * 2.0)
        for s in [self.player] + self.enemies:
            if s.ko_effect_start_ms <= 0:
                continue
            t = pygame.time.get_ticks() - s.ko_effect_start_ms
            if t >= 1000:
                # End of effect
                s.ko_effect_start_ms = 0
                continue
            # Ease pop in (0-200ms), hold, then pop out 50% faster (~867-1000ms)
            if t < 200:
                k = t / 200.0
            elif t > 867:
                k = max(0.0, (1000 - t) / 133.0)
            else:
                k = 1.0
            scale = 0.6 + 0.4 * k  # from 60% to 100%
            size = int(base_size * scale)
            size = max(1, size)
            scaled = pygame.transform.smoothscale(effect_img, (size, size))
            # Slight diagonal rotation
            rotated = pygame.transform.rotozoom(scaled, -20, 1.0)
            # Slight bob
            bob = int(4 * k)
            half_w = rotated.get_width() // 2
            half_h = rotated.get_height() // 2
            cx = s.ko_effect_anchor_px[0]
            cy = s.ko_effect_anchor_px[1] - bob
            # Clamp to remain inside world bounds
            cx = max(self.world_rect.left + half_w, min(self.world_rect.right - half_w, cx))
            cy = max(self.world_rect.top + half_h, min(self.world_rect.bottom - half_h, cy))
            pos = (cx - half_w, cy - half_h)
            self.screen.blit(rotated, pos)

    # -------------------- Input --------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if self.state == "MENU":
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if hasattr(self, "_menu_button_rect") and self._menu_button_rect.collidepoint(event.pos):
                    # Start 2s flash animation before leaving menu
                    self.menu_start_triggered_ms = pygame.time.get_ticks()
                    self.menu_flash_start_button = True
                    if getattr(self.assets, 'sfx_revive', None):
                        try:
                            self.assets.sfx_revive.play()
                        except Exception:
                            pass
                elif hasattr(self, "_menu_dev_button_rect") and self._menu_dev_button_rect.collidepoint(event.pos):
                    # 20-second test arena
                    self.menu_start_triggered_ms = pygame.time.get_ticks()
                    self.menu_flash_start_button = False
                    if getattr(self.assets, 'sfx_revive', None):
                        try:
                            self.assets.sfx_revive.play()
                        except Exception:
                            pass
                    # Flag dev test by setting a temp field
                    self._menu_pending_duration_ms = 20 * 1000
                elif hasattr(self, "_menu_grid_button_rect") and self._menu_grid_button_rect.collidepoint(event.pos):
                    self.show_menu_grid = not self.show_menu_grid
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                # No-op in menu
                pass
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
                self.menu_start_triggered_ms = pygame.time.get_ticks()
                self.menu_flash_start_button = True
                if getattr(self.assets, 'sfx_revive', None):
                    try:
                        self.assets.sfx_revive.play()
                    except Exception:
                        pass
        elif self.state == "GAME_OVER":
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.state = "MENU"
                self.menu_start_triggered_ms = 0
                if hasattr(self, '_menu_pending_duration_ms'):
                    delattr(self, '_menu_pending_duration_ms')
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN, pygame.K_SPACE):
                self.state = "MENU"
                self.menu_start_triggered_ms = 0
                if hasattr(self, '_menu_pending_duration_ms'):
                    delattr(self, '_menu_pending_duration_ms')
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
                elif event.key == pygame.K_ESCAPE:
                    self._pause_game()
        elif self.state == "PAUSED":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._resume_with_countdown()
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if hasattr(self, "_pause_continue_rect") and self._pause_continue_rect.collidepoint(event.pos):
                    self._resume_with_countdown()
                elif hasattr(self, "_pause_menu_rect") and self._pause_menu_rect.collidepoint(event.pos):
                    # Return to main menu and clear any pending auto-start
                    self.state = "MENU"
                    self.menu_start_triggered_ms = 0
                    if hasattr(self, '_menu_pending_duration_ms'):
                        delattr(self, '_menu_pending_duration_ms')
        elif self.state == "PAUSE_COUNTDOWN":
            # Ignore inputs during countdown except ESC to return to PAUSED
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = "PAUSED"


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
    # Global music: loop main theme across all screens
    try:
        pygame.mixer.init()
        music_path = os.path.join(base_dir, "assets", "sounds", "MainTheme.mp3")
        if os.path.exists(music_path):
            pygame.mixer.music.load(music_path)
            pygame.mixer.music.set_volume(0.54)
            pygame.mixer.music.play(-1)
    except Exception:
        # Audio is optional; continue without music if unavailable
        pass

    grid_cells = 25
    # Fixed world 600x600 (24px tiles), UI 120px tall, window 600x720
    tile_size = 24
    world_h = tile_size * grid_cells  # 600
    win_w = world_h
    win_h = world_h + 120
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


