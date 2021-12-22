from strategies import MinimalEngine
import chess
import random


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


class HangingEngine(MinimalEngine):
    def __init__(self, *args, name=None):
        super().__init__(*args)

    def search(self, board, time_limit, ponder):
        legal_moves = set(board.legal_moves)
        candidate_moves = []
        for target in chess.SQUARES:
            target_piece = board.piece_at(target)
            if target_piece and target_piece.color != board.turn:
                target_value = PIECE_VALUES[target_piece.piece_type]
                attackers = board.attackers(board.turn, target)
                defenders = board.attackers(not board.turn, target)

                attack_values = [PIECE_VALUES[board.piece_at(x).piece_type] for x in attackers]
                defend_values = [PIECE_VALUES[board.piece_at(x).piece_type] for x in defenders]

                if attackers and (min(attack_values) < target_value or
                                  len(defenders) < len(attackers)):
                    lowest_attacker = min(attack_values)
                    for attacker, value in zip(attackers, attack_values):
                        move = chess.Move(attacker, target)
                        if move in legal_moves:
                            candidate_moves.append(move)

        if len(candidate_moves) == 0:
            candidate_moves = list(legal_moves)

        # print([board.san(move) for move in candidate_moves])

        return random.choice(candidate_moves)


class RandomEngine(MinimalEngine):
    def __init__(self, *args, name=None):
        super().__init__(*args)

    def search(self, board, time_limit, ponder):
        candidate_moves = list(board.legal_moves)
        return random.choice(candidate_moves)


if __name__ == "__main__":
    from LearningEngine import print_game, play_game
    import sys

    hanger = HangingEngine(None, None, sys.stderr)

    random_engine = RandomEngine(None, None, sys.stderr)

    board = chess.Board('r2b4/5kp1/n5R1/1b5p/2p5/5N2/K7/8 b - - 2 40')#r1bqk1nr/1ppppp1p/7b/p5P1/3nP3/4B3/PPP2PPR/RN1QKBN1 w Qkq - 2 7')

    hanger.search(board, 1000, True)

    board = chess.Board()

    play_game(hanger, random_engine, board)

    print_game(board, "Hanging Engine", "Random")
