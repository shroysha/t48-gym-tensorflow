import gym
from gym import spaces

from .t48 import T48Game, T48Board


class T48GymEnv(gym.Env):
    GYM_ENV_NAME = "T48GymEnv-v0"

    def __init__(self):
        self.t48_game = T48Game()
        self.observation_space = spaces.Box(low=T48Board.EMPTY_TILE_VALUE,
                                            high=T48Board.MAX_TILE_VALUE,
                                            shape=self.t48_game.t48_board.shape,
                                            dtype=T48Board.TILE_DTYPE)
        self.action_space = spaces.Discrete(4)
        self._last_rewarded_score = 0

    def reset(self):
        self.t48_game.restart_game()
        self._last_rewarded_score = 0
        return self.t48_game.t48_board.board_data

    def step(self, action):
        reward = 0

        info = None
        hit_wall = False
        try:
            print(action)
            self.t48_game.do_swipe_choice(action.__int__())
        except AssertionError:
            info = "Cannot swipe"
            hit_wall = True
            reward = -1

        print("Last High Score:", self._last_rewarded_score,
              "Score:", self.t48_game.score,
              "NewHighScore:", self.has_new_high_score)

        if self.has_new_high_score:
            reward = self.t48_game.score
            self._last_rewarded_score = self.t48_game.score

        observation = self.t48_game.t48_board.board_data
        done = self.t48_game.is_game_over or (hit_wall and reward != 2)
        return observation, reward, done, info

    @property
    def has_new_high_score(self):
        return self.t48_game.score > self._last_rewarded_score

    def render(self, mode='human'):
        print("2048 :: Score :: ", str(self.t48_game.score))
        print(repr(self.t48_game.t48_board.board_data))
        print("Enter command (wsad): ")


gym.register(T48GymEnv.GYM_ENV_NAME, entry_point=T48GymEnv)
