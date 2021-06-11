from strategies import MinimalEngine
import chess
import numpy as np
from math import inf
import sys
import codecs
import json


class LearningEngine(MinimalEngine):

    def __init__(self, *args, name=None, weights=None, weight_file=None):
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

            # todo: remove this. This is just for curiosity
            #P: 24.99, N: 13.89, B: 16.21, R: 24.96, Q: 9.19, K: 0.00, M: 25.00
            self.weights = np.array([25, 13.9, 16.2, 24.96, 9.19, 0, 25])
        else:
            self.weights = weights

    @staticmethod
    def features(board):
        """
        Returns a numerical vector describing the board position
        """

        all_pieces = board.piece_map().values()

        features = np.zeros(7)

        index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

        for piece in all_pieces:
            value = -1 if piece.color == board.turn else 1

            features[index[piece.piece_type]] += value

        if board.is_checkmate():
            features[6] = 1

        return features

    def search(self, board, time_limit, ponder):

        moves = list(board.legal_moves)

        scores = np.zeros(len(moves))

        for i, move in enumerate(moves):
            # apply the current candidate move
            board.push(move)

            # calculate score
            descriptor = self.features(board)
            scores[i] = self.weights.dot(descriptor)

            board.pop()

        probs = np.exp(scores)
        probs /= np.sum(probs)

        samples = np.random.multinomial(1, probs)
        sampled_move = moves[np.min(np.argwhere(samples))]

        return sampled_move

    def save_weights(self, filepath):
        weight_list = self.weights.tolist()
        with codecs.open(filepath, 'w', encoding='utf-8') as fopen:
            json.dump(weight_list, fopen)


if __name__ == "__main__":
    engine_white = LearningEngine(None, None, sys.stderr)
    board = chess.Board()

    engine_white.weights[0] = 1.0

    engine_black = LearningEngine(None, None, sys.stderr)

    engine_black.weights = np.array([1., 3., 3., 5., 9., 0., 25.])
    engine_white.weights = np.array([1., 1., 1., 1., 1., 0., 25.])

    wins = 0
    losses = 0
    draws = 0

    max_moves = 100

    while True:
        board = chess.Board()

        white_positions = []
        black_positions = []

        while not board.outcome() and board.fullmove_number < max_moves:
            if board.turn == chess.WHITE:
                black_positions.append(engine_white.features(board))
                move = engine_white.search(board, 1000, True)
            else:
                white_positions.append(engine_black.features(board))
                move = engine_black.search(board, 1000, True)

            #print(board.san(move))
            board.push(move)

        outcome = board.outcome(claim_draw=True)

        learning_rate = 0.001

        if outcome and outcome.winner == chess.WHITE:
            print("White wins")

            # learn from white's "winning" positions
            for position in white_positions:
                engine_white.weights += learning_rate * position
            # learn from black's "losing" positions
            for position in black_positions:
                engine_white.weights -= learning_rate * position

            wins += 1

        elif outcome and outcome.winner == chess.BLACK:
            print("Black wins")

            # learn from white's "losing" positions
            for position in white_positions:
                engine_white.weights -= learning_rate * position
            # learn from black's "winning" positions
            for position in black_positions:
                engine_white.weights += learning_rate * position

            losses += 1

        else:
            print("Draw")

            # learn from white's "drawn" positions
            for position in white_positions:
                engine_white.weights -= learning_rate * position
            # learn from black's "drawn" positions
            for position in black_positions:
                engine_white.weights -= learning_rate * position

            draws += 1

        # engine_white.weights[0] = 1
        engine_white.weights = engine_white.weights.clip(min=0, max=25)

        weights = engine_white.weights

        print("P: {:.2f}, N: {:.2f}, B: {:.2f}, R: {:.2f}, Q: {:.2f}, K: {:.2f}, M: {:.2f}".format(
            weights[0], weights[1], weights[2], weights[3], weights[4], weights[5],
            weights[6]))

        print("Wins: {}. Losses: {}. Draws: {}. Win rate: {:.2f}, W/L: {:.2f}".format(
            wins, losses, draws, wins / (wins + losses + draws), wins / (losses + 1e-16)))

        engine_white.save_weights("weights.json")

