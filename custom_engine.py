from strategies import MinimalEngine
import random
import chess
import sys
import time
from math import inf as INFINITY
from collections import namedtuple

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_count(new_board, turn):
    # count material in the new position

    all_pieces = new_board.piece_map().values()

    material_difference = 0

    for piece in all_pieces:
        value = PIECE_VALUES[piece.piece_type]
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

    score += space * 1/64

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


Config = namedtuple("Config", ['prune', 'cache', 'sort', 'max_depth'], defaults=[True, True, True, 4])


def minimax_score(board, turn, opponent_best=INFINITY, my_best=-INFINITY, curr_depth=0,
                  cache=(), config=Config()):

    global cache_hits, num_pruned, positions

    positions += 1

    if curr_depth == config.max_depth or board.outcome():
        return improved_score(board, turn)

    # recursively reason about best move

    moves = list(board.legal_moves)
    best_move = None
    best_score = -INFINITY
    
    children = []

    # generate children positions from legal moves
    for move in moves:
        # apply the current candidate move

        new_board = board.copy()
        new_board.push(move)
        
        sort_score = material_count(new_board, not turn) if config.sort else 0

        children.append((sort_score, new_board, move))

    for _, new_board, move in sorted(children, key=lambda x: x[0], reverse=True):

        if config.cache:
            # The cache saves score and depth of score calculation.

            fen = new_board.fen()
            fen = fen[:-4]  # remove move counts from fen

            score, cached_depth = cache[fen] if fen in cache else (0, 0)

            # depth of score estimate if we compute it
            new_depth = config.max_depth - curr_depth

            # if we could get a deeper estimate than what is in the cache
            if new_depth > cached_depth:
                score = minimax_score(new_board, not turn, -my_best, -opponent_best, curr_depth + 1, cache, config)

                cache[fen] = (score, new_depth)
            else:
                cache_hits += 1
        else:
            score = minimax_score(new_board, not turn, -my_best, -opponent_best, curr_depth + 1, cache, config)

        if score > best_score:
            best_move = move
            best_score = score
            my_best = max(best_score, my_best)

        if config.prune:
            if score > opponent_best:
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
        self.known_positions = {}

    def search(self, board, time_limit, ponder):
        moves = list(board.legal_moves)

        best_move = None
        best_score = -INFINITY

        for move in moves:
            # apply the current candidate move

            new_board = board.copy()
            new_board.push(move)

            score = self.score_function(new_board, board.turn, cache=self.known_positions,
                                        config=self.config, curr_depth=1)

            if score > best_score:
                best_move = move
                best_score = score

        return best_move


if __name__ == "__main__":
    board = chess.Board('8/5Qpk/B4bnp/8/3r4/PR4PK/1P3P1P/6r1 b - - 2 31')
    # board = chess.Board('3rk3/1p2qp2/2p2n2/1B3bp1/1b1Qp3/8/PPPP1PP1/RNB1K1N1 w Q - 0 23')

    configs = [Config(prune=False, cache=False, sort=False),
               #Config(prune=False, cache=True, sort=False),
               # Config(prune=True, cache=False, sort=False),
               #Config(prune=True, cache=True, sort=False),
               # Config(prune=True, cache=False, sort=True),
               #Config(prune=True, cache=True, sort=True),
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

        print("\n\n\n")
