import unittest
import chess
from custom_engine import ScoreEngine, material_count, tiebreakers
import sys
import math
import time
from LearningEngine import LearningEngine
import numpy as np

missing_rook_white = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w Qkq - 0 1'
missing_rook_black = 'rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1'
queen_mate = '8/8/8/8/8/2Q2K1k/8/8 w - - 0 1'
mate_in_two = '8/6Q1/8/8/7k/4K3/8/8 w - - 2 2'
hanging_queen = '8/8/7Q/7k/5K2/8/8/8 b - - 0 1'
bad_check = '8/8/Q7/1P5k/8/8/6q1/1K6 w - - 0 1'
opening_error = 'r1bqkb1r/pppn1ppp/3Pp3/4n2Q/8/3B4/PPPP1PPP/RNB1K1NR w KQkq - 1 7'
opening_error_flipped = 'r1bqkb1r/pppn1ppp/3Pp3/4n2Q/8/3B4/PPPP1PPP/RNB1K1NR b KQkq - 1 7'

imbalanced_material_white = 'r1b1k1nr/pppppppp/8/8/8/8/PPP1PPPP/1NBQKBN1 w kq - 0 1'
imbalanced_material_black = 'r1b1k1nr/pppppppp/8/8/8/8/PPP1PPPP/1NBQKBN1 b kq - 0 1'



class EngineTestCase(unittest.TestCase):

    def test_material_count(self):
        board = chess.Board(missing_rook_white)
        self.assertEqual(material_count(board), 5)

        board = chess.Board(missing_rook_black)
        self.assertEqual(material_count(board), -5)

    def test_quiet_position(self):
        board = chess.Board(missing_rook_black)
        engine = ScoreEngine(None, None, sys.stderr)

        score = engine.quiescence_search(board)

        self.assertEqual(material_count(board), score)

    def test_queen_mate(self):
        board = chess.Board(queen_mate)
        # depth 1
        engine = ScoreEngine(None, None, sys.stderr, max_depth=1)
        move = engine.search(board, math.inf, True)
        self.assertEqual(board.san(move), 'Qh8#')

        # depth 2
        engine = ScoreEngine(None, None, sys.stderr, max_depth=2)
        move = engine.search(board, math.inf, True)
        self.assertEqual(board.san(move), 'Qh8#')

    def test_mate_in_two(self):
        board = chess.Board(mate_in_two)

        # depth 2
        engine = ScoreEngine(None, None, sys.stderr, max_depth=2)
        move = engine.search(board, math.inf, True)
        self.assertEqual(board.san(move), 'Kf4')

        # depth 3
        engine = ScoreEngine(None, None, sys.stderr, max_depth=3)
        move = engine.search(board, math.inf, True)
        self.assertEqual(board.san(move), 'Kf4')

    def material_q_move(self, fen, true_material, true_quiet, true_move, depth):
        board = chess.Board(fen)

        material = material_count(board)
        print("Material count is {}".format(material))
        self.assertEqual(material, true_material)

        engine = ScoreEngine(None, None, sys.stderr, max_depth=depth)

        # quiescence search
        qs = engine.quiescence_search(board)
        print("Quiescence search found the score was {}".format(qs))
        self.assertEqual(qs, true_quiet)

        if true_move:
            move = engine.search(board, math.inf, True)
            self.assertEqual(board.san(move), true_move)

    def test_hanging_queen(self):
        self.material_q_move(hanging_queen, 9, 0, 'Kxh6', 2)

    def test_bad_check(self):
        self.material_q_move(bad_check, -1, -1, None, 2)

    # def test_opening_error(self):
    #     board = chess.Board(opening_error)
    #     engine = ScoreEngine(None, None, sys.stderr, max_depth=3)
    #
    #     move = engine.search(board, math.inf, True)
    #
    #     print(board.san(move))

    def test_caching_speed(self):
        board = chess.Board(opening_error)
        engine = ScoreEngine(None, None, sys.stderr, max_depth=3)

        start = time.time()
        qs1 = engine.quiescence_search(board)
        qs1_time = time.time() - start

        print("1st quiescence search took {} seconds.".format(qs1_time))

        start = time.time()
        qs2 = engine.quiescence_search(board)
        qs2_time = time.time() - start

        print("2nd quiescence search took {} seconds.".format(qs2_time))

        start = time.time()
        move1 = engine.search(board, math.inf, True)
        search1_time = time.time() - start

        self.assertEqual(qs1, qs2)
        self.assertGreater(qs1_time, qs2_time)

        print("1st move search took {} seconds.".format(search1_time))

        start = time.time()
        move2 = engine.search(board, math.inf, True)
        search2_time = time.time() - start

        print("2nd move search took {} seconds.".format(search2_time))

        self.assertEqual(move1, move2)
        self.assertGreater(search1_time, search2_time)

    def test_tiebreaker(self):
        board = chess.Board(opening_error)

        score = tiebreakers(board)

        print("Tiebreaker score is {}".format(score))

    def test_learning_features(self):
        board = chess.Board(imbalanced_material_white)

        engine = LearningEngine(None, None, sys.stderr)

        features = engine.features(board)

        flipped_board = chess.Board(imbalanced_material_black)

        flipped_features = engine.features(flipped_board)

        print("Features for white to move:")
        print(features)
        print("Features for black to move:")
        print(flipped_features)

        self.assertTrue(np.allclose(features, -flipped_features))

    def test_weight_saving(self):
        engine = LearningEngine(None, None, sys.stderr)
        print("Initialized engine with weights")
        print(engine.weights)

        path = "weights.json"  # todo: make this a proper temp file
        engine.save_weights(path)

        new_engine = LearningEngine(None, None, sys.stderr, weight_file=path)
        print("New engine loaded with weights")
        print(new_engine.weights)

        self.assertTrue(np.allclose(new_engine.weights, engine.weights))

if __name__ == '__main__':
    unittest.main()
