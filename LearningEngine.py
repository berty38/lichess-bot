import copy
import random
import sys
import time
from collections import deque
from datetime import datetime
import tempfile

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from HangingEngine import HangingEngine
from strategies import MinimalEngine, RandomMove

DEFAULT_MODEL_LOCATION = "/Users/bert/Desktop/latest_model"

EVAL_INTERVAL = 50
OPPONENT_UPDATE_INTERVAL = 50
NUM_STEPS = 500

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

PIECE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}


def material_count(new_board):
    # count material in the new position for player to move

    all_pieces = new_board.piece_map().values()

    material_difference = 0

    if new_board.is_checkmate():
        if new_board.outcome().winner == new_board.turn:
            material_difference += 100
        else:
            material_difference -= 100

    for piece in all_pieces:
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == new_board.turn:
            material_difference += value
        else:
            material_difference -= value

    return material_difference


def get_reward(board, next_board):
    return -material_count(next_board) - material_count(board)


def print_game(board, white_name, black_name):
    to_print = list()
    to_print.append(f'[FEN "{board.starting_fen}"]\n[White "{white_name}"]\n[Black "{black_name}"]\n')
    
    new_board = chess.Board(board.starting_fen)
    san_moves = []
    for move in board.move_stack:
        san_moves += [new_board.san(move)]
        new_board.push(move)

    for i in range(len(san_moves)):
        if i % 2 == 0:
            to_print.append("%d." % (i / 2 + 1))
        to_print.append(san_moves[i])

    return " ".join(to_print)


def play_game(a, b, board=None, print_pgn=False, writer=None, step=None):
    if not board:
        board = chess.Board()
    max_moves = 200
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
        pgn = print_game(board, white.engine.id["name"], black.engine.id["name"])

        if writer:
            writer.add_text("PGN", pgn, step)
        else:
            print(pgn)

    if outcome and outcome.winner == a_turn:
        return 1, board, a_turn
    elif outcome and outcome.winner == (not a_turn):
        return -1, board, a_turn
    return 0, board, a_turn


def play_match(a, b, num_games, writer=None, step=None, offset=0, starting_position=None):
    """
    Play a match between engine a and b
    """
    wins = 0
    losses = 0
    draws = 0
    total_score = 0
    for i in range(num_games):
        if writer:
            game_step = step * 1000 + i + offset
        if starting_position:
            board = chess.Board(starting_position)
            board.starting_fen = starting_position
        else:
            board = None
        result, final_board, a_turn = play_game(a, b, print_pgn=True, writer=writer, step=game_step, board=board)
        if result == -1:
            losses += 1
        elif result == 0:
            draws += 1
        elif result == 1:
            wins += 1
        if a_turn == final_board.turn:
            total_score += material_count(final_board)
        else:
            total_score -= material_count(final_board)

    return wins, losses, draws, total_score / num_games


def get_state_action_features(board, move):
    old_count = material_count(board)
    board.push(move)
    board_features = get_features(board)
    new_count = material_count(board)
    board.pop()

    return np.concatenate((board_features, np.array((old_count, new_count))))


def get_features(board):
    """
    Returns a numpy vector describing the board position
    """

    # mirror board if black's turn
    if board.turn == chess.BLACK:
        board = board.mirror()

    all_pieces = board.piece_map().items()

    piece_count = np.zeros(7)

    # add castling rights
    castling = [board.has_kingside_castling_rights(chess.WHITE),
                board.has_queenside_castling_rights(chess.WHITE),
                board.has_kingside_castling_rights(chess.BLACK),
                board.has_queenside_castling_rights(chess.BLACK)]

    piece_grid = np.zeros((64, 12))

    for position, piece in all_pieces:
        value = -1 if piece.color == board.turn else 1

        type_index = PIECE_INDEX[piece.piece_type]

        piece_count[type_index] += value

        if piece.color != board.turn:
            type_index += 6

        piece_grid[position, type_index] = 1

    # if board.is_checkmate():
        # piece_count[6] = 1
    piece_count[6] = material_count(board)

    return np.concatenate((piece_count, castling, piece_grid.ravel()))


class Net(nn.Module):
    def __init__(self, d, hidden_sizes=[128, 64, 32, 16]):
        super().__init__()

        self.first = nn.Linear(d, hidden_sizes[0])
        self.hidden_fc = nn.Linear(hidden_sizes[-1], 1)
        self.data_fc = nn.Linear(d, 1)

        hidden_layers = []
        for i in range(len(hidden_sizes) - 1):
            hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x0):
        if x0.ndim == 1:
            # convert to batch view
            x0 = x0.view((1, -1))
        h = functional.leaky_relu(self.first(x0))
        for layer in self.hidden_layers:
            h = functional.leaky_relu(layer(h))
        return self.hidden_fc(h) + self.data_fc(x0)


class TorchLearner:
    def __init__(self, buffer_size=75000, batch_size=25, sync_interval=10000,
                 from_file=DEFAULT_MODEL_LOCATION):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.sync_interval = sync_interval

        blank_board = chess.Board()
        e4 = chess.Move.from_uci("e2e4")

        sample_vector = get_state_action_features(blank_board, e4)

        self.online_net = torch.jit.script(Net(sample_vector.size))
        self.target_net = torch.jit.script(Net(sample_vector.size))
        self.online_net.eval()
        self.target_net.eval()
        self.loss_fn = functional.huber_loss

        self.optimizer = torch.optim.Adam(self.online_net.parameters())

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, verbose=True, patience=50)

        if from_file:
            self.online_net.load_state_dict(torch.load(from_file))
            self.target_net.load_state_dict(torch.load(from_file))

        self.update_count = 0

    def score(self, board, move):
        self.online_net.eval()
        with torch.no_grad():
            return self.online_net(torch.from_numpy(get_state_action_features(board, move).astype(np.float32)))

    def store_transition(self, reward, prev_board, prev_move, new_board):
        # store position in buffer
        prev_features = get_state_action_features(prev_board, prev_move).astype(np.float32)
        new_boards = []
        for move in new_board.legal_moves:
            new_boards.append(get_state_action_features(new_board, move).astype(np.float32))
        # if len(self.buffer) < self.buffer.maxlen:
        self.buffer.append((float(reward), prev_features, new_boards))

    def learn(self):
        return np.mean([self._q_learn() for _ in range(NUM_STEPS)])

    def _q_learn(self):
        discount = 0.9

        # sample a batch
        batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))

        rewards, prev_features, new_boards = zip(*batch)

        rewards = torch.from_numpy(np.asarray(rewards, dtype=np.float32))

        prev_features = torch.from_numpy(np.asarray(prev_features))

        new_boards = [torch.from_numpy(np.asarray(x)) for x in new_boards]
        
        self.online_net.train()
        self.optimizer.zero_grad()

        # todo: find a better way to do this than a loop
        new_scores = []
        for position_new_boards in new_boards:
            if len(position_new_boards) > 0:
                    with torch.no_grad():
                        next_scores = self.online_net(position_new_boards)
                        best_index = torch.argmax(next_scores)

                        new_score = self.target_net(position_new_boards[best_index])
                        # standard q-learning
                        # next_scores = self.online_net(position_new_boards)
                        # new_score = torch.max(next_scores)

                        new_scores.append(new_score)
            else:
                new_scores.append(0)  # this should only happen in terminal states

        new_scores = torch.FloatTensor(new_scores)
        prev_scores = self.online_net(prev_features)

        loss = self.loss_fn(prev_scores.view(len(batch)),
                            rewards - discount * new_scores)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1)

        if len(self.buffer) > self.batch_size:
            self.optimizer.step()
            self.update_count += 1
            
        self.online_net.eval()

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
        score = self.learner.score(board, move)
        return score

    def search(self, board, time_limit, ponder):
        # returns a random choice among highest-scoring q values
        moves = np.array(list(board.legal_moves))

        scores = np.zeros(len(moves))

        for i, move in enumerate(moves):
            # apply the current candidate move
            scores[i] = self.action_score(board, move)

        # consider scores within 0.25 of max to be equivalent

        best_moves = moves[scores.max() - scores < 0.1]

        return np.random.choice(best_moves)


def main():
    from torch.utils.tensorboard import SummaryWriter

    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    note = "Minimax_Q_learning_256_Adam_Large_batch_Leaky_ReLU_XSteps"

    # base_dir = tempfile.TemporaryDirectory().name
    base_dir = "/Users/bert/Desktop"
    log_dir = '{}/logs/'.format(base_dir)
    print("Storing logs in {}".format(log_dir))
    writer = SummaryWriter(log_dir + note + time_string, flush_secs=1)

    step = 0
    start_time = time.time()

    # Use this next line to load bot weights from disk
    #engine_learner = LearningEngine(None, None, sys.stderr, learner=TorchLearner(from_file=DEFAULT_MODEL_LOCATION))
    # use this one to re-initialize
    engine_learner = LearningEngine(None, None, sys.stderr, learner=TorchLearner(from_file=None))

    opponent_learner = LearningEngine(None, "LearningEngine{}".format(step), sys.stderr,
                                      learner=copy.deepcopy(engine_learner.learner))
    opponent_random = RandomMove(None, None, sys.stderr)
    opponent_hanging = HangingEngine(None, None, sys.stderr)

    opponent_wins = np.zeros(3)
    opponent_losses = np.zeros(3)
    opponent_draws = np.zeros(3)

    result_log = deque(maxlen=OPPONENT_UPDATE_INTERVAL)

    max_moves = 1000
    eps_reset = 0

    blank_board = chess.Board()
    e4 = chess.Move.from_uci("e2e4")

    sample_vector = get_state_action_features(blank_board, e4)

    writer.add_graph(engine_learner.learner.online_net,
                     torch.Tensor(sample_vector))
    writer.flush()

    buffer_full = False

    hang_score = 0
    learn_score = 0.5
    rand_score = 0.5
    q_loss_record = deque(maxlen=1000)

    while True:
        # update engine name
        engine_learner.engine.id["name"] = "LearningEngine{}".format(step)

        board = chess.Board()

        learner_color = np.random.rand() < 0.5
        game_positions = []
        game_moves = []

        # choose opponent
        opponents = [opponent_random, opponent_hanging, opponent_learner]
        # choose opponent that we are worst against
        opponent_scores = [rand_score, hang_score, learn_score]

        opponent_index = opponent_scores.index(min(opponent_scores))

        engine_opponent = opponents[opponent_index]

        # play a single game
        while not board.outcome() and board.fullmove_number < max_moves:
            game_positions.append(board.copy(stack=False))

            # play a game
            if board.turn == learner_color:
                epsilon = max(0.05, 1 / np.sqrt((step - eps_reset) / 10 + 100))

                # use epsilon-greedy strategy
                if np.random.rand() < epsilon:
                    move = np.random.choice(list(board.legal_moves))
                else:
                    move = engine_learner.search(board, 1000, True)
            else:
                move = engine_opponent.search(board, 1000, True)

            game_moves.append(move)

            # print(board.san(move))
            board.push(move)

        game_positions.append(board.copy(stack=False))

        # do learning on all steps of the game
        outcome = board.outcome()
        rewards = []
        q_losses = []
        learner_reward = 0

        # do q-learning on each moves (including opponents)
        for i in range(len(game_moves)):
            reward = get_reward(game_positions[i], game_positions[i + 1])
            rewards.append(reward)

            if game_positions[i].turn == learner_color:
                learner_reward += reward
            else:
                learner_reward -= reward

            engine_learner.learner.store_transition(reward, game_positions[i], game_moves[i], game_positions[i + 1])

        # final move
        if outcome and outcome.winner == learner_color:
            opponent_wins[opponent_index] += 1
            result_log.append(1)
        elif outcome and outcome.winner == (not learner_color):
            opponent_losses[opponent_index] += 1
            result_log.append(-1)
        else:
            opponent_draws[opponent_index] += 1
            result_log.append(0)

        buffer_full = engine_learner.learner.buffer.maxlen == len(engine_learner.learner.buffer)

        q_losses = engine_learner.learner.learn()

        # log diagnostic info
        if step % EVAL_INTERVAL == 0:
            num_games = 10
            rand_wins, rand_losses, rand_ties = play_match(engine_learner, opponent_random, num_games)
            learn_wins, learn_losses, learn_ties = play_match(engine_learner, opponent_learner, num_games)
            hang_wins, hang_losses, hang_ties = play_match(engine_learner, opponent_hanging, num_games)

            rand_score = rand_wins + 0.5 * rand_ties
            learn_score = learn_wins + 0.5 * learn_ties
            hang_score = hang_wins + 0.5 * hang_ties

            writer.add_scalar("Win Rate v. Random", rand_wins / num_games, step)
            writer.add_scalar("Loss Rate v. Random", rand_losses / num_games, step)
            writer.add_scalar("Win Rate v. Hanging", hang_wins / num_games, step)
            writer.add_scalar("Loss Rate v. Hanging", hang_losses / num_games, step)
            writer.add_scalar("Win Rate v. Learner", learn_wins / num_games, step)
            writer.add_scalar("Loss Rate v. Learner", learn_losses / num_games, step)
            writer.add_scalar("Score v. Learner", learn_score, step)
            writer.add_scalar("Score v. Hanging", hang_score, step)
            writer.add_scalar("Score v. Random", rand_score, step)
            torch.save(engine_learner.learner.online_net.state_dict(), DEFAULT_MODEL_LOCATION)

        writer.add_scalar("Reward", learner_reward, step)
        writer.add_scalar("Training Result", result_log[-1], step)

        # if buffer_full:
        writer.add_scalar("Avg Loss", np.mean(q_losses), step)

        q_loss_record.append(np.mean(q_losses))

        # if step > 1000:
            # engine_learner.learner.scheduler.step(np.mean(q_loss_record))

        if step > 1 and step % OPPONENT_UPDATE_INTERVAL == 0 and np.mean(result_log) > 0.5:
            print("Updating opponent to current model parameters")
            # update opponent engine
            opponent_learner = LearningEngine(None, "LearningEngine{}".format(step), sys.stderr,
                                              learner=copy.deepcopy(engine_learner.learner))

            opponent_learner.engine.id["name"] = "LearningEngine{}".format(step)

        step += 1

        elapsed_time = time.time() - start_time

        print("Played {} games ({:.2f} games/sec)".format(step, step / elapsed_time))
        if not buffer_full:
            print("Buffer is {} full".format(len(engine_learner.learner.buffer) / engine_learner.learner.buffer.maxlen))


if __name__ == "__main__":
    main()
