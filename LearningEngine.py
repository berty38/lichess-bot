import copy
import random
import sys
import time
from collections import deque
from datetime import datetime

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from HangingEngine import HangingEngine
from strategies import MinimalEngine, RandomMove

DEFAULT_MODEL_LOCATION = "/Users/bert/Desktop/latest_model"

EVAL_INTERVAL = 50

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


def print_game(board, white_name, black_name):

    print('[White "{}"]'.format(white_name))
    print('[Black "{}"]'.format(black_name))

    new_board = chess.Board()
    san_moves = []
    for move in board.move_stack:
        san_moves += [new_board.san(move)]
        new_board.push(move)

    to_print = []

    for i in range(len(san_moves)):
        if i % 2 == 0:
            to_print.append("%d." % (i / 2 + 1))
        to_print.append(san_moves[i])

    print(" ".join(to_print))


def play_game(a, b, board=None, print_pgn=False):
    if not board:
        board = chess.Board()
    max_moves = 60
    a_turn = random.getrandbits(1)  # random boolean
    while not board.outcome(claim_draw=True) and board.fullmove_number < max_moves:

        # play a game
        if board.turn == a_turn:
            move = a.search(board, 10000, True)
        else:
            move = b.search(board, 10000, True)

        board.push(move)

    outcome = board.outcome()

    if print_pgn:
        white = a if a_turn else b
        black = b if a_turn else a
        print_game(board, white.engine.id["name"], black.engine.id["name"])

    if outcome and outcome.winner == a_turn:
        return 1
    elif outcome and outcome.winner == (not a_turn):
        return -1
    return 0


def play_match(a, b, num_games):
    """
    Play a match between engine a and b
    """
    wins = 0
    losses = 0
    draws = 0
    for _ in range(num_games):
        result = play_game(a, b, print_pgn=True)
        if result == -1:
            losses += 1
        elif result == 0:
            draws += 1
        elif result == 1:
            wins += 1

    return wins, losses, draws


def get_features(board):
    """
    Returns a numpy vector describing the board position
    """

    # mirror board if black's turn
    if board.turn == chess.BLACK:
        board = board.mirror()

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

    # add castling rights
    castling = [board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK)]

    piece_grid = np.zeros((64, 12))

    for position, piece in all_pieces:
        value = -1 if piece.color == board.turn else 1

        type_index = index[piece.piece_type]

        features[type_index] += value

        if piece.color != board.turn:
            type_index += 6

        piece_grid[position, type_index] = 1

    if board.is_checkmate():
        features[6] = 1

    return np.concatenate((features, castling, piece_grid.ravel()))


class Net(nn.Module):
    def __init__(self, d, layers):
        super().__init__()
        self.mid_layers = nn.ModuleList([torch.nn.Linear(d, d) for _ in range(layers - 1)])
        self.fc = torch.nn.Linear(d, 1)

    def forward(self, x0):
        x = x0
        for layer in self.mid_layers:
            x = layer(x)
            x = functional.relu(x) + x0
        return self.fc(x)


class TorchLearner:
    def __init__(self, buffer_size=10000, batch_size=5, sync_interval=200, resnet_depth=2,
                 from_file=DEFAULT_MODEL_LOCATION):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.sync_interval = sync_interval

        sample_vector = get_features(chess.Board())

        self.online_net = Net(sample_vector.size, resnet_depth)
        self.target_net = Net(sample_vector.size, resnet_depth)
        self.loss_fn = functional.l1_loss

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), amsgrad=True)

        if from_file:
            self.online_net.load_state_dict(torch.load(from_file))
            self.target_net.load_state_dict(torch.load(from_file))

        self.update_count = 0

    def score(self, board):
        with torch.no_grad():
            return self.online_net(torch.from_numpy(get_features(board).astype(np.float32)))

    def learn(self, reward, prev_board, prev_move, new_board):
        # q(a, s) is estimate of discounted future reward after
        #       making move a from s
        # q(a, s) <- q(a, s) + learning_rate *
        #                       (reward +  discount * max_{a'} q(a', s') - q(a, s))

        # store position in buffer
        prev_board.push(prev_move)
        prev_features = get_features(prev_board).astype(np.float32)
        prev_board.pop()
        new_boards = []
        for move in new_board.legal_moves:
            new_board.push(move)
            new_boards.append(get_features(new_board).astype(np.float32))
            new_board.pop()
        self.buffer.append((float(reward), prev_features, new_boards))

        return self._q_learn()

    def _q_learn(self):
        discount = 0.9

        if len(self.buffer) < self.batch_size:
            return 0

        self.optimizer.zero_grad()

        # sample a batch
        batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))

        rewards, prev_features, new_boards = zip(*batch)

        rewards = torch.from_numpy(np.asarray(rewards, dtype=np.float32))

        prev_features = torch.from_numpy(np.asarray(prev_features))

        new_boards = [torch.from_numpy(np.asarray(x)) for x in new_boards]

        # todo: find a better way to do this than a loop
        new_scores = []
        for position_new_boards in new_boards:
            if len(position_new_boards) > 0:
                with torch.no_grad():
                    # get best action
                    next_scores = self.online_net(position_new_boards)
                    best_index = torch.argmax(next_scores)

                    # compute target
                    new_score = self.target_net(position_new_boards[best_index])

                new_scores.append(new_score)
            else:
                new_scores.append(0)  # this should only happen in terminal states

        new_scores = torch.FloatTensor(new_scores)
        prev_scores = self.online_net(prev_features)

        loss = self.loss_fn(prev_scores.view(self.batch_size), rewards + discount * new_scores)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1)

        self.optimizer.step()

        for param in self.online_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.update_count += 1

        if self.update_count % self.sync_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.data


class LearningEngine(MinimalEngine):

    def __init__(self, *args, name=None, learner=None):

        if learner is None:
            learner = TorchLearner(from_file=DEFAULT_MODEL_LOCATION)

        super().__init__(*args)
        self.name = name
        self.learner = learner

    def action_score(self, board, move):
        # calculate score
        board.push(move)
        score = self.learner.score(board)
        board.pop()
        return score

    def search(self, board, time_limit, ponder):
        # returns a random choice among highest-scoring q values
        moves = np.array(list(board.legal_moves))

        scores = np.zeros(len(moves))

        for i, move in enumerate(moves):
            # apply the current candidate move
            scores[i] = self.action_score(board, move)

        best_moves = moves[scores == scores.max()]

        return np.random.choice(best_moves)

    def learn(self, reward, prev_board, prev_move, new_board):
        loss = self.learner.learn(reward, prev_board, prev_move, new_board)
        return loss


def main():
    from tensorflow import summary

    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    note = "All_Opponents_10000_buffer"

    # base_dir = tempfile.TemporaryDirectory().name
    base_dir = "/Users/bert/Desktop"
    log_dir = '{}/logs/'.format(base_dir)
    print("Storing logs in {}".format(log_dir))
    writer = summary.create_file_writer(log_dir + note + time_string)

    step = 0
    start_time = time.time()

    # Use this next line to load bot weights from disk
    # engine_learner = LearningEngine(None, None, sys.stderr, learner=TorchLearner(from_file=DEFAULT_MODEL_LOCATION))
    # use this one to re-initialize
    engine_learner = LearningEngine(None, None, sys.stderr, learner=TorchLearner(from_file=None))

    opponent_learner = LearningEngine(None, "LearningEngine{}".format(step), sys.stderr,
                                      learner=copy.deepcopy(engine_learner.learner))
    opponent_random = RandomMove(None, None, sys.stderr)
    opponent_hanging = HangingEngine(None, None, sys.stderr)

    opponent_wins = np.zeros(3)
    opponent_losses = np.zeros(3)
    opponent_draws = np.zeros(3)

    max_moves = 60
    eps_reset = 0

    while True:
        # update engine name
        engine_learner.engine.id["name"] = "LearningEngine{}".format(step)

        board = chess.Board()

        learner_color = np.random.rand() < 0.5
        learner_positions = []
        learner_moves = []

        # choose opponent
        opponents = [opponent_random, opponent_hanging, opponent_learner]
        engine_opponent = random.choice(opponents)
        # engine_opponent = opponent_learner

        opponent_index = opponents.index(engine_opponent)

        # play a single game
        while not board.outcome() and board.fullmove_number < max_moves:
            # play a game
            if board.turn == learner_color:
                learner_positions.append(board.copy(stack=False))

                epsilon = max(0.01, 1 / np.sqrt((step - eps_reset) / 10 + 100))

                # use epsilon-greedy strategy
                if np.random.rand() < epsilon:
                    move = np.random.choice(list(board.legal_moves))
                else:
                    move = engine_learner.search(board, 1000, True)
                learner_moves.append(move)
            else:
                move = engine_opponent.search(board, 1000, True)

            # print(board.san(move))
            board.push(move)

        # do learning on all steps of the game
        outcome = board.outcome()
        rewards = []
        q_losses = []

        # do q-learning on each of white's moves
        for i in range(len(learner_moves) - 1):
            reward = material_count(learner_positions[i + 1]) - \
                     material_count(learner_positions[i])
            rewards.append(reward)

            q_loss = engine_learner.learn(reward, learner_positions[i], learner_moves[i],
                                          learner_positions[i + 1])
            q_losses.append(q_loss)

        # final move
        if outcome and outcome.winner == learner_color:
            reward = 100
            opponent_wins[opponent_index] += 1
        elif outcome and outcome.winner == (not learner_color):
            reward = -100
            opponent_losses[opponent_index] += 1
        else:
            reward = 0
            opponent_draws[opponent_index] += 1

        rewards.append(reward)

        episode_reward = np.sum(rewards)

        q_loss = engine_learner.learn(reward, learner_positions[-1], learner_moves[-1],
                                      chess.Board('8/8/8/8/8/8/8/8 w - - 0 1'))
        q_losses.append(q_loss)

        # log diagnostic info
        if step % EVAL_INTERVAL == 0:
            with writer.as_default():
                num_games = 10
                rand_wins, rand_losses, rand_ties = play_match(engine_learner, opponent_random, num_games)
                learn_wins, learn_losses, learn_ties = play_match(engine_learner, opponent_learner, num_games)
                hang_wins, hang_losses, hang_ties = play_match(engine_learner, opponent_hanging, num_games)

                summary.scalar("Win Rate v. Random", rand_wins / num_games, step)
                summary.scalar("Loss Rate v. Random", rand_losses / num_games, step)
                summary.scalar("Win Rate v. Hanging", hang_wins / num_games, step)
                summary.scalar("Loss Rate v. Hanging", hang_losses / num_games, step)
                summary.scalar("Win Rate v. Learner", learn_wins / num_games, step)
                summary.scalar("Loss Rate v. Learner", learn_losses / num_games, step)
            print("Updating opponent to current model parameters")
            # update opponent engine
            opponent_learner = LearningEngine(None, "LearningEngine{}".format(step), sys.stderr,
                                              learner=copy.deepcopy(engine_learner.learner))

            opponent_learner.engine.id["name"] = "LearningEngine{}".format(step)

            torch.save(engine_learner.learner.online_net.state_dict(), DEFAULT_MODEL_LOCATION)

        with writer.as_default():
            summary.scalar("Reward", episode_reward, step)
            summary.scalar("Avg Loss", np.mean(q_losses), step)

        step += 1

        elapsed_time = time.time() - start_time

        print("Played {} games ({:.2f} games/sec)".format(step, step / elapsed_time))


if __name__ == "__main__":
    main()
