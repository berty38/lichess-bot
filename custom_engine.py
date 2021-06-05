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


def material_count(new_board):
    # count material in the new position for player who just moved

    all_pieces = new_board.piece_map().values()

    material_difference = 0

    for piece in all_pieces:
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == new_board.turn:
            material_difference -= value
        else:
            material_difference += value

    return material_difference


num_pruned = 0
cache_hits = 0
positions = 0


class ScoreEngine(MinimalEngine):

    def __init__(self, *args, name=None, max_depth=6):
        super().__init__(*args)
        self.name = name
        self.known_positions = {}
        self.visited_positions = set()
        self.max_depth = max_depth

    def cached_score(self, new_board):
        """
        Compute the raw evaluation of a position
        or use a cached negamax score if available
        """
        key = new_board._transposition_key()

        if key in self.known_positions:
            score, _ = self.known_positions[key]
            return score
        return material_count(new_board)

    def store_position(self, board):
        """
        Store actually visited position. If a position has been visited before,
        flag it for potential 3-fold repetition by zeroing its cache value
        """
        key = board._transposition_key()

        if key in self.visited_positions:
            self.known_positions[key] = (0, INFINITY)
        else:
            self.visited_positions.add(key)

    def get_all_moves(self, board, moves):
        """
        Return a scored list of moves to search over
        """
        children = []

        # generate children positions from legal moves
        for move in moves:
            board.push(move)  # apply the current candidate move
            sort_score = self.cached_score(board)
            board.pop()  # undo the candidate move

            children.append((sort_score, move))

        return children

    def loud_moves_only(self, board, moves):
        """
        Return a scored list of moves to search over
        """
        children = []

        was_check = board.is_check()

        # generate children positions from legal moves
        for move in moves:

            is_capture = board.is_capture(move)

            # check if move is a capture or check
            board.push(move)  # apply the current candidate move

            if was_check or board.is_check() or is_capture:
                sort_score = self.cached_score(board)
                children.append((sort_score, move))

            board.pop()  # undo the candidate move

        # if children:
        #     print(board.fen())
        #     print("Found {} loud moves.".format(len(children)))

        return children

    def quiescence_search(self, board):
        key = board._transposition_key()
        if key in self.known_positions:
            score, _ = self.known_positions[key]
        else:
            score = self.negamax_score(board, curr_depth=1, deadline=None,
                                  generate_children=self.loud_moves_only,
                                  evaluation_function=self.cached_score,
                                  caching=False, early_stop=True, max_depth=2)
            self.known_positions[key] = (score, 0)
        return score

    def negamax_score(self, board, opponent_best=INFINITY, my_best=-INFINITY,
                      curr_depth=0, max_depth=4, deadline=None, generate_children=get_all_moves,
                      evaluation_function=material_count, caching=True, early_stop=False):

        global cache_hits, num_pruned, positions

        positions += 1

        turn = board.turn

        # with claim_draw=False, outcome will not know about repetition, but we handle this elsewhere
        outcome = board.outcome(claim_draw=False)

        if outcome:
            if outcome.winner is None:
                return 0
            else:
                return 10000 / curr_depth  # prefer shallower checkmates

        if curr_depth == max_depth:
            # if we are at a terminal node, return the raw score or cached score
            return evaluation_function(board)

        # recursively reason about best move

        moves = list(board.legal_moves)
        best_move = None
        best_score = -INFINITY

        if early_stop and not board.is_check():
            best_score = -evaluation_function(board)

        children = generate_children(board, moves)

        if len(children) == 0:
            # this should only happen with quiescence search
            return evaluation_function(board)

        for _, move in sorted(children, key=lambda x: x[0], reverse=True):

            board.push(move)

            if deadline and time.time() > deadline:
                score = self.cached_score(board)
            else:
                # The cache saves score and depth of score calculation.

                key = board._transposition_key()

                score, cached_depth = self.known_positions[key] \
                    if key in self.known_positions else (0, 0)

                # depth of score estimate if we compute it
                new_depth = max_depth - curr_depth

                # if we could get a deeper estimate than what is in the cache
                if new_depth > cached_depth or not caching:
                    score = self.negamax_score(board, -my_best, -opponent_best, curr_depth + 1,
                                               max_depth, deadline, generate_children, evaluation_function)

                    self.known_positions[key] = (score, new_depth)
                else:
                    cache_hits += 1

            board.pop()

            if score > best_score:
                best_move = move
                best_score = score
                my_best = max(best_score, my_best)

            if score > opponent_best:
                num_pruned += 1
                return -best_score

        return -best_score

    def search(self, board, time_limit, ponder):
        print("Searching with time limit {} and ponder {}, turn is {}".format(time_limit, ponder, board.turn))
        # store current position

        if isinstance(time_limit, chess.engine.Limit):
            target_time = time_limit.time
        else:
            # target 50 moves
            remaining = max(15, 40 - board.fullmove_number)
            target_time = time_limit / remaining / 1000
        print("Trying to make move in {} seconds".format(target_time))
        deadline = time.time() + target_time

        self.store_position(board)

        moves = list(board.legal_moves)

        for depth in range(1, self.max_depth + 1):

            print("Trying depth {}".format(depth))

            best_moves = []
            best_score = -INFINITY

            for move in moves:
                # apply the current candidate move

                new_board = board.copy()
                new_board.push(move)

                score = self.negamax_score(new_board, curr_depth=1, max_depth=depth,
                                           deadline=deadline,
                                           generate_children=self.get_all_moves,
                                           evaluation_function=self.quiescence_search)

                if score > best_score:
                    best_moves = [move]
                    best_score = score
                elif score == best_score:
                    best_moves.append(move)

            print("Found {} moves with score {}".format(len(best_moves), best_score))

            if deadline and time.time() > deadline:
                print("Ran out of time at depth {}".format(depth))
                break

        best_move = random.choice(best_moves)

        # store new position
        board.push(best_move)
        self.store_position(board)
        board.pop()

        return best_move


if __name__ == "__main__":
    board = chess.Board('8/5Qpk/B4bnp/8/3r4/PR4PK/1P3P1P/6r1 b - - 2 31')
    #board = chess.Board('3rk3/1p2qp2/2p2n2/1B3bp1/1b1Qp3/8/PPPP1PP1/RNB1K1N1 w Q - 0 23')
    #board = chess.Board('rk6/8/3n2b1/8/4P3/1B6/5N2/1K6 b - - 0 1')
    # # obvious mate for white
    # board = chess.Board('r3kbnr/pppppppp/8/8/8/8/PPPQPPPP/1NBRKBNR w Kkq - 0 1')
    # # obvious mate for black
    #board = chess.Board('8/8/8/Q4q2/8/8/7r/2K5 b - - 0 1')

    # todo: refactor to keep stats without global variables
    cache_hits = 0
    num_pruned = 0
    positions = 0

    engine = ScoreEngine(None, None, sys.stderr)

    start_time = time.time()

    score = engine.search(board, 999999, True)

    print("Found move in {} seconds".format(time.time() - start_time))

    print("Cache hits: {}. Prunes: {}. Positions: {}.".format(cache_hits, num_pruned, positions))

    print("Score = {}".format(score))

    print("\n")
