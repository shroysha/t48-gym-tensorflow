import gym
from gym import spaces

from .t48 import T48Game, T48Board


class T48GymEnv(gym.Env):

    GYM_ENV_NAME="T48GymEnv-v0"

    def __init__(self):
        self.t48_game = T48Game()
        self.observation_space = spaces.Box(low=T48Board.EMPTY_TILE_VALUE,
                                            high=T48Board.MAX_TILE_VALUE,
                                            shape=self.t48_game.t48_board.shape,
                                            dtype=T48Board.TILE_DTYPE)
        self.action_space = spaces.Discrete(4)

    def step(self, action):
        self.t48_game.do_swipe_choice(action)

        observation = self.t48_game.t48_board.board_data
        reward = self.t48_game.score
        done = self.t48_game.is_game_over
        info = None
        return observation, reward, done, info

    def reset(self):
        self.t48_game.restart_game()
        return self.t48_game.t48_board.board_data

    def render(self, mode='human'):
        print("2048 :: Score :: ", str(self.t48_game.score))
        print(repr(self.t48_game.t48_board.board_data))
        print("Enter command (wsad): ")


gym.register(T48GymEnv.GYM_ENV_NAME, entry_point=T48GymEnv)

