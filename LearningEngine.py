from strategies import MinimalEngine
import chess
import numpy as np
from math import inf
import sys


class LearningEngine(MinimalEngine):

    def __init__(self, *args, name=None, weights=None):
        super().__init__(*args)
        self.name = name

        if weights is None:
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


if __name__ == "__main__":
    engine_white = LearningEngine(None, None, sys.stderr)
    board = chess.Board()

    engine_white.weights[0] = 1.0

    engine_black = LearningEngine(None, None, sys.stderr)

    wins = 0
    losses = 0
    draws = 0

    while True:
        # # randomly catch black up
        # if np.random.rand() < 0.1:
        #     engine_black.weights = engine_white.weights.copy()

        board = chess.Board()

        white_positions = []
        black_positions = []

        while not board.outcome(claim_draw=True):
            if board.turn == chess.WHITE:
                move = engine_white.search(board, 1000, True)
                white_positions.append(engine_white.features(board))
            else:
                move = engine_black.search(board, 1000, True)
                black_positions.append(engine_black.features(board))

            #print(board.san(move))
            board.push(move)

        outcome = board.outcome(claim_draw=True)

        learning_rate = 0.001

        if outcome.winner == chess.WHITE:
            print("White wins")

            for position in white_positions:
                engine_white.weights += learning_rate * position

            wins += 1

        elif outcome.winner == chess.BLACK:
            print("Black wins")

            for position in white_positions:
                engine_white.weights -= learning_rate * position

            losses += 1

        else:
            print("Draw")

            for position in white_positions:
                engine_white.weights -= learning_rate * position

            draws += 1

        engine_white.weights[0] = 1
        engine_white.weights = engine_white.weights.clip(min=0, max=25)

        weights = engine_white.weights

        print("P: {:.2f}, N: {:.2f}, B: {:.2f}, R: {:.2f}, Q: {:.2f}, K: {:.2f}, M: {:.2f}".format(
            weights[0], weights[1], weights[2], weights[3], weights[4], weights[5],
            weights[6]))

        print("Wins: {}. Losses: {}. Draws {}".format(wins, losses, draws))


