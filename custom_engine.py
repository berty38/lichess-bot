from strategies import MinimalEngine
import random
import chess

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

    # remove hanging pieces from material count

    all_pieces = new_board.piece_map().items()

    for square, piece in all_pieces:
        if piece.color == turn:
            attacker_count = len(new_board.attackers(not turn, square))
            defender_count = len(new_board.attackers(turn, square))
            if attacker_count > defender_count:
                score -= PIECE_VALUES[piece.symbol().upper()]

    return score


def minimax_score(board, turn, curr_depth=0, max_depth=2):

    if curr_depth == max_depth or board.outcome():
        return material_count(board, turn)

    # recursively reason about best move

    moves = list(board.legal_moves)
    best_move = None
    best_score = -float('inf')  # todo: make this not ugly

    for move in moves:
        # apply the current candidate move

        new_board = board.copy()
        new_board.push(move)

        score = minimax_score(new_board, not turn,
                              curr_depth + 1, max_depth)

        if score > best_score:
            best_move = move
            best_score = score

    # print("Opponent's best move is {}".format(best_move))

    return -best_score


class ScoreEngine(MinimalEngine):

    def __init__(self, *args, name=None):
        super().__init__(*args)
        self.name = name
        self.score_function = minimax_score

    def search(self, board, time_limit, ponder):
        moves = list(board.legal_moves)

        best_move = None
        best_score = -float('inf')  # todo: make this not ugly

        for move in moves:
            # apply the current candidate move

            new_board = board.copy()
            new_board.push(move)

            score = self.score_function(new_board, board.turn)

            if score > best_score:
                best_move = move
                best_score = score

        return best_move
