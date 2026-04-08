"""
env.py — Custom Gymnasium environment for Flappy Bird RL agent.

Contains:
  - FlappyBirdEnv: Core Gymnasium environment with physics, pipes, rewards
  - NoisyObservationWrapper: Adds Gaussian sensor noise to observations
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Try importing pygame for optional rendering
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

BIRD_X = 80          # Fixed horizontal position of the bird
BIRD_RADIUS = 15

PIPE_WIDTH = 60
PIPE_SPEED = 2.5     # Pixels per step (base)
PIPE_SPAWN_INTERVAL = 120  # Steps between new pipes

BASE_GRAVITY = 0.5
FLAP_IMPULSE = -8.0
MAX_VELOCITY = 10.0

BASE_GAP_SIZE = 150  # Vertical gap between top and bottom pipe


# ---------------------------------------------------------------------------
# FlappyBirdEnv
# ---------------------------------------------------------------------------
class FlappyBirdEnv(gym.Env):
    """
    A Flappy Bird–like Gymnasium environment.

    Observation (4,):
        [bird_y_norm, bird_velocity_norm, dist_to_next_pipe_norm, pipe_gap_center_y_norm]

    Actions:
        0 — do nothing
        1 — flap

    Rewards:
        +0.1  per timestep survived
        +1.0  for passing a pipe
        -10.0 for collision (episode terminates)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        # Spaces
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Pygame state (initialised lazily)
        self.screen = None
        self.clock = None

        # Episode state — set in reset()
        self.bird_y = 0.0
        self.bird_vel = 0.0
        self.pipes = []
        self.score = 0
        self.steps = 0
        self.gravity = BASE_GRAVITY
        self.gap_size = BASE_GAP_SIZE

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Domain randomization
        self.gravity = BASE_GRAVITY + self.np_random.uniform(-0.05, 0.05)
        self.gap_size = BASE_GAP_SIZE + self.np_random.uniform(-20, 20)

        # Bird state
        self.bird_y = SCREEN_HEIGHT / 2.0
        self.bird_vel = 0.0

        # Pipes: list of dicts {x, gap_center_y, scored}
        self.pipes = []
        self._spawn_pipe(initial_x=SCREEN_WIDTH + 100)

        self.score = 0
        self.steps = 0

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"

        # --- Physics ---
        if action == 1:
            self.bird_vel = FLAP_IMPULSE
        self.bird_vel += self.gravity
        self.bird_vel = np.clip(self.bird_vel, -MAX_VELOCITY, MAX_VELOCITY)
        self.bird_y += self.bird_vel

        # --- Move pipes & spawn ---
        for pipe in self.pipes:
            pipe["x"] -= PIPE_SPEED

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if p["x"] + PIPE_WIDTH > 0]

        # Spawn new pipe when the last one has travelled far enough
        if len(self.pipes) == 0 or self.pipes[-1]["x"] < SCREEN_WIDTH - PIPE_SPAWN_INTERVAL:
            self._spawn_pipe()

        # --- Reward / Termination ---
        reward = 0.1  # survival bonus
        terminated = False
        truncated = False

        # Dense reward: bonus for being vertically close to the gap center
        next_pipe = self._get_next_pipe()
        if next_pipe is not None:
            dist_to_gap = abs(self.bird_y - next_pipe["gap_center_y"])
            max_dist = SCREEN_HEIGHT / 2.0
            alignment_bonus = 0.5 * (1.0 - dist_to_gap / max_dist)
            reward += max(0.0, alignment_bonus)

        # Check scoring (bird passed a pipe)
        for pipe in self.pipes:
            if not pipe["scored"] and pipe["x"] + PIPE_WIDTH < BIRD_X:
                pipe["scored"] = True
                self.score += 1
                reward += 1.0

        # Check collision
        if self._check_collision():
            reward = -10.0
            terminated = True

        self.steps += 1

        obs = self._get_obs()
        info = {"score": self.score}

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _spawn_pipe(self, initial_x=None):
        """Add a new pipe at the right edge of the screen."""
        x = initial_x if initial_x is not None else SCREEN_WIDTH + PIPE_WIDTH
        gap_center = self.np_random.uniform(
            self.gap_size / 2 + 40,
            SCREEN_HEIGHT - self.gap_size / 2 - 40,
        )
        self.pipes.append({"x": float(x), "gap_center_y": float(gap_center), "scored": False})

    def _get_next_pipe(self):
        """Return the nearest pipe that hasn't been fully passed yet."""
        for pipe in sorted(self.pipes, key=lambda p: p["x"]):
            if pipe["x"] + PIPE_WIDTH > BIRD_X:
                return pipe
        # Fallback (shouldn't happen normally)
        return self.pipes[-1] if self.pipes else None

    def _get_obs(self):
        """Construct a normalised observation vector."""
        next_pipe = self._get_next_pipe()
        if next_pipe is None:
            dist_to_pipe = 1.0
            gap_center_y = 0.5
        else:
            dist_to_pipe = (next_pipe["x"] - BIRD_X) / SCREEN_WIDTH
            gap_center_y = next_pipe["gap_center_y"] / SCREEN_HEIGHT

        bird_y_norm = self.bird_y / SCREEN_HEIGHT
        bird_vel_norm = (self.bird_vel + MAX_VELOCITY) / (2 * MAX_VELOCITY)

        obs = np.array(
            [
                np.clip(bird_y_norm, 0.0, 1.0),
                np.clip(bird_vel_norm, 0.0, 1.0),
                np.clip(dist_to_pipe, 0.0, 1.0),
                np.clip(gap_center_y, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return obs

    def _check_collision(self):
        """Return True if the bird has collided with pipes, ground, or ceiling."""
        # Ground / ceiling
        if self.bird_y - BIRD_RADIUS <= 0 or self.bird_y + BIRD_RADIUS >= SCREEN_HEIGHT:
            return True

        # Pipe collision (axis-aligned bounding box)
        for pipe in self.pipes:
            px = pipe["x"]
            gap_top = pipe["gap_center_y"] - self.gap_size / 2
            gap_bot = pipe["gap_center_y"] + self.gap_size / 2

            # Is bird horizontally overlapping the pipe?
            if BIRD_X + BIRD_RADIUS > px and BIRD_X - BIRD_RADIUS < px + PIPE_WIDTH:
                # Is bird outside the gap?
                if self.bird_y - BIRD_RADIUS < gap_top or self.bird_y + BIRD_RADIUS > gap_bot:
                    return True

        return False

    # ------------------------------------------------------------------
    # Rendering (pygame)
    # ------------------------------------------------------------------
    def _render_frame(self):
        if not PYGAME_AVAILABLE:
            return None

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Flappy Bird — RL Agent")
            else:
                self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()

        # Background
        self.screen.fill((135, 206, 235))  # sky blue

        # Ground
        pygame.draw.rect(self.screen, (222, 184, 135), (0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 20))

        # Pipes
        for pipe in self.pipes:
            gap_top = pipe["gap_center_y"] - self.gap_size / 2
            gap_bot = pipe["gap_center_y"] + self.gap_size / 2
            px = int(pipe["x"])
            # Top pipe
            pygame.draw.rect(self.screen, (34, 139, 34), (px, 0, PIPE_WIDTH, int(gap_top)))
            # Bottom pipe
            pygame.draw.rect(
                self.screen,
                (34, 139, 34),
                (px, int(gap_bot), PIPE_WIDTH, SCREEN_HEIGHT - int(gap_bot)),
            )

        # Bird
        pygame.draw.circle(
            self.screen, (255, 255, 0), (int(BIRD_X), int(self.bird_y)), BIRD_RADIUS
        )

        # Score text
        font = pygame.font.SysFont(None, 36)
        score_surf = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_surf, (10, 10))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


# ---------------------------------------------------------------------------
# NoisyObservationWrapper
# ---------------------------------------------------------------------------
class NoisyObservationWrapper(gym.ObservationWrapper):
    """
    Adds Gaussian noise to observations to simulate noisy sensors.

    Parameters
    ----------
    env : gym.Env
        The environment to wrap.
    noise_std : float
        Standard deviation of the Gaussian noise (default 0.02).
    """

    def __init__(self, env, noise_std=0.02):
        super().__init__(env)
        self.noise_std = noise_std

    def observation(self, obs):
        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=obs.shape).astype(
            np.float32
        )
        noisy_obs = obs + noise
        # Clip to stay within observation space bounds
        noisy_obs = np.clip(noisy_obs, self.observation_space.low, self.observation_space.high)
        return noisy_obs
