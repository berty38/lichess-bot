from strategies import MinimalEngine
import random
import chess
import sys
import time
from math import inf as INFINITY
from collections import namedtuple

PIECE_VALUES = {
    'P': 1,
    'N': 3,
    'B': 3,
    'R': 5,
    'Q': 9,
    'K': 0,
}


def material_count(new_board, turn):
    # count material in the new position

    all_pieces = new_board.piece_map().values()

    material_difference = 0

    for piece in all_pieces:
        value = PIECE_VALUES[piece.symbol().upper()]
        if piece.color == turn:
            material_difference += value
        else:
            material_difference -= value

    if new_board.is_checkmate():
        material_difference += 999999

    return material_difference


def improved_score(new_board, turn):
    score = material_count(new_board, turn)

    # add extra score strategies

    # compute space controlled by current color

    space = 0
    for square in chess.SQUARES:
        if new_board.is_attacked_by(turn, square):
            space += 1
        if new_board.is_attacked_by(not turn, square):
            space -= 1

    score += space * 1/32

    # # remove hanging pieces from material count
    #
    # all_pieces = new_board.piece_map().items()
    #
    # for square, piece in all_pieces:
    #     if piece.color == turn:
    #         attacker_count = len(new_board.attackers(not turn, square))
    #         defender_count = len(new_board.attackers(turn, square))
    #         if attacker_count > defender_count:
    #             score -= PIECE_VALUES[piece.symbol().upper()]

    return score


num_pruned = 0
cache_hits = 0
positions = 0


Config = namedtuple("Config", ['prune', 'cache', 'sort', 'max_depth'], defaults=[True, True, False, 3])


def minimax_score(board, turn, cutoff=INFINITY, curr_depth=0, cache=(), config=Config()):

    global cache_hits, num_pruned, positions

    positions += 1

    if curr_depth == config.max_depth or board.outcome():
        return improved_score(board, turn)

    # recursively reason about best move

    moves = list(board.legal_moves)
    best_move = None
    best_score = -INFINITY

    for move in moves:
        # apply the current candidate move

        new_board = board.copy()
        new_board.push(move)

        if config.cache:
            fen = new_board.fen()
            if fen not in cache:
                score = minimax_score(new_board, not turn, -best_score, curr_depth + 1, cache, config)

                cache[fen] = score
            else:
                score = cache[fen]
                cache_hits += 1
        else:
            score = minimax_score(new_board, not turn, -best_score, curr_depth + 1, cache, config)

        if score > best_score:
            best_move = move
            best_score = score

        if config.prune:
            if score > cutoff:
                num_pruned += 1
                return -best_score

    # print("Opponent's best move is {}".format(best_move))

    return -best_score


class ScoreEngine(MinimalEngine):

    def __init__(self, *args, name=None, config=Config()):
        super().__init__(*args)
        self.name = name
        self.score_function = minimax_score
        self.config = config

    def search(self, board, time_limit, ponder):
        moves = list(board.legal_moves)

        best_move = None
        best_score = -INFINITY

        known_positions = {}

        for move in moves:
            # apply the current candidate move

            new_board = board.copy()
            new_board.push(move)

            score = self.score_function(new_board, board.turn, cache=known_positions, config=self.config, curr_depth=1)

            if score > best_score:
                best_move = move
                best_score = score

        return best_move


if __name__ == "__main__":
    # board = chess.Board('8/5Qpk/B4bnp/8/3r4/PR4PK/1P3P1P/6r1 b - - 2 31')
    board = chess.Board('3rk3/1p2qp2/2p2n2/1B3bp1/1b1Qp3/8/PPPP1PP1/RNB1K1N1 w Q - 0 23')

    configs = [Config(prune=False, cache=False),
               Config(prune=False, cache=True),
               Config(prune=True, cache=False),
               Config(prune=True, cache=True),
               ]

    for config in configs:
        # todo: refactor to keep stats without global variables
        cache_hits = 0
        num_pruned = 0
        positions = 0

        print("Starting " + repr(config))

        engine = ScoreEngine(None, None, sys.stderr, config=config)
        start_time = time.time()
        move = engine.search(board, time_limit=999, ponder=False)
        print("Found move in {} seconds".format(time.time() - start_time))

        print("Cache hits: {}. Prunes: {}. Positions: {}.".format(cache_hits, num_pruned, positions))

        print(move)
