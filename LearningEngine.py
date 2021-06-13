from strategies import MinimalEngine
import chess
import numpy as np
from math import inf
import sys
import codecs
import json


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_count(new_board):
    # count material in the new position for player to move

    all_pieces = new_board.piece_map().values()

    material_difference = 0

    for piece in all_pieces:
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == new_board.turn:
            material_difference += value
        else:
            material_difference -= value

    return material_difference


class LearningEngine(MinimalEngine):

    def __init__(self, *args, name=None, weights=None, weight_file="weights.json"):
        super().__init__(*args)
        self.name = name

        if weight_file:
            # load weights from file
            with codecs.open(weight_file, 'r', encoding='utf-8') as fopen:
                weight_test = fopen.read()
                weight_list = json.loads(weight_test)
            self.weights = np.array(weight_list)
        elif weights is None:
            # initialize weights
            starting_board = chess.Board()
            descriptor = self.features(starting_board)
            self.weights = np.random.rand(descriptor.size)
        else:
            self.weights = weights

    @staticmethod
    def features(board):
        """
        Returns a numerical vector describing the board position
        """

        all_pieces = board.piece_map().items()

        features = np.zeros(7)

        index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

        piece_grid = np.zeros((64, 6))

        for position, piece in all_pieces:
            value = -1 if piece.color == board.turn else 1

            type_index = index[piece.piece_type]

            features[type_index] += value

            piece_grid[position, type_index] = value

        if board.is_checkmate():
            features[6] = 1

        return np.concatenate((features, piece_grid.ravel()))

    def action_score(self, board, move):
        # calculate score
        board.push(move)
        descriptor = self.features(board)
        board.pop()

        score = self.weights.dot(descriptor)

        return score

    def search(self, board, time_limit, ponder):

        moves = list(board.legal_moves)

        scores = np.zeros(len(moves))

        for i, move in enumerate(moves):
            # apply the current candidate move

            scores[i] = self.action_score(board, move)

        probs = np.exp(scores - scores.max())
        probs /= np.sum(probs)

        samples = np.random.multinomial(1, probs)
        sampled_move = moves[np.min(np.argwhere(samples))]

        return sampled_move

    def save_weights(self, filepath):
        weight_list = self.weights.tolist()
        with codecs.open(filepath, 'w', encoding='utf-8') as fopen:
            json.dump(weight_list, fopen)

    def q_learn(self, reward, prev_board, prev_move, new_board, learning_rate=0.0001, discount=1.0):
        # q(a, s) is estimate of discounted future reward after
        #       making move a from s
        # q(a, s) <- q(a, s) + learning_rate *
        #                       (reward +  max_{a'} q(a', s') - q(a, s))
        # weights <- weights + learning_rate *
        #                       (reward +

        moves = list(new_board.legal_moves)
        if len(moves) == 0:
            max_future_score = 0
        else:
            max_future_score = max([self.action_score(new_board, move) for move in moves])

        prev_board.push(prev_move)
        descriptor = self.features(prev_board)
        prev_board.pop()

        self.weights += learning_rate * (reward + discount * max_future_score -
                                         self.action_score(prev_board, prev_move)) * descriptor


if __name__ == "__main__":
    engine_white = LearningEngine(None, None, sys.stderr)
    board = chess.Board()

    engine_black = LearningEngine(None, None, sys.stderr, weights=None, weight_file=None)

    engine_black.weights[:7] = [1., 3., 3., 5., 9., 0., 25.]
    engine_black.weights[7:] = 0
    # engine_white.weights[:7] = [1., 3., 3., 5., 9., 0., 25.]
    # engine_white.weights[7:] = 0

    wins = 0
    losses = 0
    draws = 0

    max_moves = 100

    while True:
        board = chess.Board()

        white_positions = []
        white_moves = []

        # play a single game

        while not board.outcome() and board.fullmove_number < max_moves:
            if board.turn == chess.WHITE:
                white_positions.append(board.copy())
                move = engine_white.search(board, 1000, True)
                white_moves.append(move)
            else:
                move = engine_black.search(board, 1000, True)

            #print(board.san(move))
            board.push(move)

        # do learning on all steps of the game

        outcome = board.outcome(claim_draw=True)

        learning_rate = 0.1

        # do q-learning on each of white's moves
        for i in range(len(white_moves) - 1):
            reward = material_count(white_positions[i + 1]) - \
                     material_count(white_positions[i])
            engine_white.q_learn(reward, white_positions[i], white_moves[i],
                                 white_positions[i + 1])

        # final move
        if outcome and outcome.winner == chess.WHITE:
            reward = 100
            wins += 1
        elif outcome and outcome.winner == chess.BLACK:
            reward = -100
            losses += 1
        else:
            reward = 0
            draws += 1

        engine_white.q_learn(reward, white_positions[-1], white_moves[-1],
                             chess.Board('8/8/8/8/8/8/8/8 w - - 0 1'))

        # clip weights
        engine_white.weights = engine_white.weights.clip(min=-25, max=25)


        # print diagnostic info
        weights = engine_white.weights
        print("P: {:.2f}, N: {:.2f}, B: {:.2f}, R: {:.2f}, Q: {:.2f}, K: {:.2f}, M: {:.2f}".format(
            weights[0], weights[1], weights[2], weights[3], weights[4], weights[5],
            weights[6]))

        print("Wins: {}. Losses: {}. Draws: {}. Win rate: {:.2f}, W/L: {:.2f}".format(
            wins, losses, draws, wins / (wins + losses + draws), wins / (losses + 1e-16)))

        engine_white.save_weights("weights.json")

