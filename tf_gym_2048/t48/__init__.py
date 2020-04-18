import random
import numpy as np
import itertools


class T48Board:

    DEFAULT_SIZE = 4
    TILE_DTYPE = np.int64
    EMPTY_TILE_VALUE = 0
    MAX_TILE_VALUE = 2048

    def __init__(self):
        self.board_size = self.DEFAULT_SIZE
        self.board_data = np.zeros(self.shape, self.TILE_DTYPE)
        self.spawn_new_tile()

    @property
    def shape(self):
        return self.board_size, self.board_size

    @property
    def xy_range(self):
        return itertools.product(range(self.board_size), repeat=2)

    def set_board_data(self, board_data):
        self.board_data = board_data.copy()

    @property
    def unfilled_tiles(self):
        return [(x, y) for x, y in self.xy_range if self.board_data[x, y] == self.EMPTY_TILE_VALUE]

    def spawn_new_tile(self):
        new_tile_value = 2

        unfilled = self.unfilled_tiles
        random_tile = unfilled[random.randint(0, len(unfilled) - 1)]

        self.board_data[random_tile[0], random_tile[1]] = new_tile_value

    @property
    def preview_swipe_left(self):
        board = self.board_data
        flipfunc = (lambda x: x)
        return T48Board.__preview_row_op(board, flipfunc)

    @property
    def preview_swipe_up(self):
        board = self.board_data.T
        flipfunc = (lambda x: x.T)
        return T48Board.__preview_row_op(board, flipfunc)

    @property
    def preview_swipe_right(self):
        board = self.board_data.T[::-1].T
        flipfunc = (lambda x: x.T[::-1].T)
        return T48Board.__preview_row_op(board, flipfunc)

    @property
    def preview_swipe_down(self):
        board = self.board_data[::-1].T
        flipfunc = (lambda x: x.T[::-1])
        return T48Board.__preview_row_op(board, flipfunc)

    def __can_swipe(self, preview_func):
        return not np.array_equal(self.board_data, preview_func)

    @property
    def can_swipe_left(self):
        return self.__can_swipe(self.preview_swipe_left)

    @property
    def can_swipe_up(self):
        return self.__can_swipe(self.preview_swipe_up)

    @property
    def can_swipe_right(self):
        return self.__can_swipe(self.preview_swipe_right)

    @property
    def can_swipe_down(self):
        return self.__can_swipe(self.preview_swipe_down)

    def __do_swipe(self, assert_func, preview_func):
        assert assert_func
        self.set_board_data(preview_func)
        self.spawn_new_tile()

    def do_swipe_left(self):
        self.__do_swipe(self.can_swipe_left, self.preview_swipe_left)

    def do_swipe_up(self):
        self.__do_swipe(self.can_swipe_up, self.preview_swipe_up)

    def do_swipe_right(self):
        self.__do_swipe(self.can_swipe_right, self.preview_swipe_right)

    def do_swipe_down(self):
        self.__do_swipe(self.can_swipe_down, self.preview_swipe_down)

    @staticmethod
    def __preview_row_op(rotated_board, flip_func):
        new_mat = list()
        for i in range(rotated_board.shape[0]):
            new_mat.append(T48Board.__condense_row_left(rotated_board[i, :]))

        return flip_func(np.array(new_mat, dtype=T48Board.TILE_DTYPE))

    @staticmethod
    def __condense_row_left(row):
        row = np.array(row)
        for i in range(len(row)):
            if row[i] != T48Board.EMPTY_TILE_VALUE:
                new_i = i
                blocked = False
                for maybe_i in range(0, i)[::-1]:
                    if row[maybe_i] == T48Board.EMPTY_TILE_VALUE:
                        new_i = maybe_i
                    elif row[maybe_i] == row[i] and not blocked:
                        new_i = maybe_i
                    else:
                        blocked = True
                if new_i != i:
                    if row[i] == row[new_i]:
                        row[new_i] *= 2
                    else:
                        row[new_i] = row[i]
                    row[i] = T48Board.EMPTY_TILE_VALUE
        return row


class T48Game:

    SWIPE_CHOICES = dict()
    SWIPE_CHOICES[0] = T48Board.do_swipe_up
    SWIPE_CHOICES["w"] = T48Board.do_swipe_up
    SWIPE_CHOICES["up"] = T48Board.do_swipe_up
    SWIPE_CHOICES[1] = T48Board.do_swipe_left
    SWIPE_CHOICES["a"] = T48Board.do_swipe_left
    SWIPE_CHOICES["left"] = T48Board.do_swipe_left
    SWIPE_CHOICES[2] = T48Board.do_swipe_down
    SWIPE_CHOICES["s"] = T48Board.do_swipe_down
    SWIPE_CHOICES["down"] = T48Board.do_swipe_down
    SWIPE_CHOICES[3] = T48Board.do_swipe_right
    SWIPE_CHOICES["d"] = T48Board.do_swipe_right
    SWIPE_CHOICES["right"] = T48Board.do_swipe_right

    def __init__(self):
        self.t48_board = T48Board()

    def do_swipe_choice(self, choice):
        assert choice in T48Game.SWIPE_CHOICES.keys()
        T48Game.SWIPE_CHOICES[choice](self.t48_board)

    @property
    def score(self):
        a_score = T48Board.EMPTY_TILE_VALUE
        for x, y in self.t48_board.xy_range:
            a_score = max(a_score, self.t48_board.board_data[x, y])
        return a_score

    def restart_game(self):
        T48Board.__init__(self.t48_board)

    @property
    def is_game_over(self):
        return not self.t48_board.can_swipe_up \
               and not self.t48_board.can_swipe_down \
               and not self.t48_board.can_swipe_left \
               and not self.t48_board.can_swipe_right

