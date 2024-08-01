import numpy as np
from typing import List, Tuple
from itertools import combinations

# Constants
BOARD_SIZE = 11
INITIAL_QUODS = 20
INITIAL_QUAZARS = 6

# Cell states
EMPTY_CELL = 0
QUOD_RED = 1
QUOD_BLUE = 2
QUAZAR = 3
DISABLED_CELL = -1

# Players
PLAYER_RED = 1
PLAYER_BLUE = 2


class QuodGame:
    def __init__(self):
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY_CELL, dtype=np.int8)
        self.quazar_owners = np.full((BOARD_SIZE, BOARD_SIZE), 0, dtype=np.int8)
        self.current_player = PLAYER_RED
        self.quods = {PLAYER_RED: INITIAL_QUODS, PLAYER_BLUE: INITIAL_QUODS}
        self.quazars = {PLAYER_RED: INITIAL_QUAZARS, PLAYER_BLUE: INITIAL_QUAZARS}
        self.remove_corners()
        self.moves = []
        self.current_turn = 1
        self.current_turn_moves = []
        self.last_move = None
        self.winning_square = None

    def remove_corners(self):
        self.board[0, 0] = self.board[0, -1] = self.board[-1, 0] = self.board[-1, -1] = DISABLED_CELL

    def make_move(self, row: int, col: int, piece_type: int) -> bool:
        if self.board[row, col] != EMPTY_CELL:
            return False
        if self.quods[self.current_player] == 0 and piece_type in [QUOD_RED, QUOD_BLUE]:
            return False
        if (self.current_player == PLAYER_RED and piece_type == QUOD_BLUE) or \
                (self.current_player == PLAYER_BLUE and piece_type == QUOD_RED):
            return False

        self.board[row, col] = piece_type
        self.last_move = (row, col)

        if piece_type in [QUOD_RED, QUOD_BLUE]:
            self.quods[self.current_player] -= 1
            self.current_turn_moves.append((self.current_player, piece_type, (row, col)))

            # Check for square formation immediately after placing a quod
            self.winning_square = self.check_for_square(self.current_player)
            if self.winning_square:
                self.end_turn()
                return True

            self.end_turn()
        elif piece_type == QUAZAR:
            if self.quazars[self.current_player] > 0:
                self.quazars[self.current_player] -= 1
                self.quazar_owners[row, col] = self.current_player
                self.current_turn_moves.append((self.current_player, piece_type, (row, col)))
            else:
                return False
        else:
            return False

        return True

    def end_turn(self):
        self.moves.append((self.current_turn, self.current_player, self.current_turn_moves.copy()))
        self.current_player = PLAYER_BLUE if self.current_player == PLAYER_RED else PLAYER_RED
        self.current_turn_moves = []
        if self.current_player == PLAYER_RED:
            self.current_turn += 1

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r, c] == EMPTY_CELL]

    def is_terminal(self) -> bool:
        return self.winning_square is not None or (self.quods[PLAYER_RED] == 0 and self.quods[PLAYER_BLUE] == 0)

    def check_for_square(self, player: int) -> List[Tuple[Tuple[int, int], ...]]:
        player_quod = QUOD_RED if player == PLAYER_RED else QUOD_BLUE
        player_pegs = np.argwhere(self.board == player_quod)
        squares = []

        if self.last_move is not None and len(player_pegs) >= 4:
            other_pegs = [tuple(peg) for peg in player_pegs if tuple(peg) != self.last_move]
            for three_pegs in combinations(other_pegs, 3):
                four_pegs = (self.last_move,) + three_pegs
                if self.is_square(four_pegs):
                    squares.append(four_pegs)

        return squares[0] if squares else None

    def is_square(self, points: List[Tuple[int, int]]) -> bool:
        def distance_sq(point1, point2):
            return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

        distances = [distance_sq(p1, p2) for p1, p2 in combinations(points, 2)]

        sorted_distances = sorted(set(distances))

        if len(sorted_distances) != 2:
            return False

        side_sq, diagonal_sq = sorted_distances

        rel_tol = 1e-9

        if not np.isclose(diagonal_sq, 2 * side_sq, rtol=rel_tol):
            return False

        side_count = sum(1 for d in distances if np.isclose(d, side_sq, rtol=rel_tol))
        diagonal_count = sum(1 for d in distances if np.isclose(d, diagonal_sq, rtol=rel_tol))

        return side_count == 4 and diagonal_count == 2

    def get_winner(self) -> int:
        if self.winning_square:
            return PLAYER_RED if self.board[
                                     self.winning_square[0][0], self.winning_square[0][1]] == QUOD_RED else PLAYER_BLUE
        elif self.quods[PLAYER_RED] == 0 and self.quods[PLAYER_BLUE] == 0:
            if self.quazars[PLAYER_RED] > self.quazars[PLAYER_BLUE]:
                return PLAYER_RED
            elif self.quazars[PLAYER_BLUE] > self.quazars[PLAYER_RED]:
                return PLAYER_BLUE
        return 0  # Draw or game not finished

    def get_square_sides(self, square_points):
        def distance_sq(p1, p2):
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

        distances = [(i, j, distance_sq(square_points[i], square_points[j]))
                     for i in range(4) for j in range(i + 1, 4)]

        distances.sort(key=lambda x: x[2])

        # The first 4 distances should be the sides
        sides = distances[:4]

        return [(square_points[side[0]], square_points[side[1]]) for side in sides]

    def __str__(self):
        symbols = {EMPTY_CELL: ' ', QUOD_RED: 'R', QUOD_BLUE: 'B', QUAZAR: 'Q', DISABLED_CELL: 'X'}
        return '\n'.join([''.join([symbols[cell] for cell in row]) for row in self.board])