import copy
import random
import sys
import time
from collections import deque, Counter
from datetime import datetime

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
import matplotlib.pyplot as plt

from HangingEngine import HangingEngine
from strategies import MinimalEngine, RandomMove
from custom_engine import ScoreEngine
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from itertools import islice

from LearningEngine import *

from math import ceil

torch.manual_seed(0)
random.seed(0)
np.random.seed(0) 

NOTE = "CNN128_MLP2048_batch50_Reg_MarginLoss_Buffer1_L2_MaxLoss"
EVAL_INTERVAL = 120  # seconds between evaluation against baselines
TARGET_UPDATE = 0  # How many steps between updates of the target model
BATCH_SIZE = 50  # number of board positions per gradient batch
MAX_MOVES = 200  # maximum number of moves in simulated games for training data
BUFFER_MAX_SIZE = 1000000  # max number of positions in buffer
EPOCHS = 100  # number of epochs before forcibly adding a new game to the buffer
NUM_GAMES = 1  # number of games to add when adding new positions
MAX_BUFFER_ROUNDS = 1  # max number of games in buffer
LR = 1e-4  # learning rate
ZERO_REGULARIZER = 0.1  # weight for zero score regularizer from initial board
WARM_START = False  # whether to load model from disk
ANCHOR_REGULARIZER = 100  # weight for labeled position regularizer
STORE_LOG = True  # whether to store tensorboard log

STARTING_POSITION = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # standard
# STARTING_POSITION = "7r/8/4k3/8/8/4K3/8/R7 w - - 0 1"  # King and rook vs king and rook
# STARTING_POSITION = "4k3/4q3/8/8/8/8/3Q4/3K4 w - - 0 1"  # King and queen vs king and queen
# STARTING_POSITION = "4k3/4q3/8/8/8/8/8/3K4 w - - 0 1"  # King and queen vs king
# STARTING_POSITION = "5rr1/1k6/8/8/7K/8/4R3/2R5 w - - 0 1"  # 4 rooks ladder mate in 2
# STARTING_POSITION = "5rr1/k7/8/8/7K/8/1R6/2R5 w - - 2 2"  # 4 rooks ladder mate in 1
# STARTING_POSITION = "Q6k/3ppppp/8/3q4/8/8/1P6/KR6 b - - 0 1"  # only legal move is mate
# STARTING_POSITION = "Q6k/4pppp/8/3q4/8/8/1P6/KR6 b - - 0 1"  # 1 legal move is mate, other is loss
# STARTING_POSITION = "8/3k4/4n3/8/8/8/2N5/1K6 w - - 0 1"  # King and knight each, no mate possible
# STARTING_POSITION = "1n2k1n1/8/8/8/8/8/8/1N2K1N1 w - - 0 1"  # King and 2 knights each. No mates possible
# STARTING_POSITION = "rn1qkbnr/ppp1pppp/3p4/8/4P1b1/P1N5/1PPP1PPP/R1BQKBNR b KQkq - 0 3"  # Hanging queen


def batchify(data, batch_size: int):
    shuffled_data = list(data)
    random.shuffle(shuffled_data)
    iterator = iter(shuffled_data)
    batches = []
    full_batches = len(data) // batch_size
    remainder = len(data) - full_batches * batch_size
    for _ in range(full_batches):
        batches.append(list(islice(iterator, batch_size)))

    if remainder:
        if batches:
            batches[-1].extend(iterator)
        else:
            batches.append(list(iterator))

    return batches


def get_features(board):
    """
    Returns a numpy vector describing the board position as a grid
    """

    # mirror board if black's turn
    if board.turn == chess.BLACK:
        def rank(s):
            return 7 - chess.square_rank(s)
    else:
        def rank(s):
            return chess.square_rank(s)

    all_pieces = board.piece_map().items()

    piece_grid = np.zeros((13, 8, 8))

    attacker_grid = np.zeros((12, 8, 8))

    for square, piece in all_pieces:
        type_index = PIECE_INDEX[piece.piece_type]

        if piece.color != board.turn:
            type_index += len(PIECE_INDEX)

        position_rank = rank(square)
        position_file = chess.square_file(square)

        piece_grid[type_index, position_rank, position_file] = 1

        for attacked_square in board.attacks(square):
            attacked_rank = rank(attacked_square)
            attacked_file = chess.square_file(attacked_square)

            attacker_grid[type_index, attacked_rank, attacked_file] += 1

    # add castling rights
    if board.has_kingside_castling_rights(board.turn):
        piece_grid[12, 0, 7] = 1
    if board.has_queenside_castling_rights(board.turn):
        piece_grid[12, 0, 0] = 1
    if board.has_kingside_castling_rights(not board.turn):
        piece_grid[12, 7, 7] = 1
    if board.has_queenside_castling_rights(not board.turn):
        piece_grid[12, 7, 7] = 1

    # add en passant squares
    ep = board.ep_square
    if ep:
        piece_grid[12, rank(ep), chess.square_file(ep)] = 1

    features = np.concatenate((piece_grid, attacker_grid), 0)

    return features


class ChessResnet(nn.Module):
    def __init__(self, depth=3, width=128):
        super().__init__()
        d = 64 * (12 + 13)
        self.in_layers = nn.ModuleList([nn.Linear(d, width) for _ in range(depth)])
        self.mid_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.final = nn.Linear(width, 1)

    def forward(self, x):
        
        if x.ndim == 3:
            x = x.view(1, -1, 8, 8)
        x = torch.flatten(x, 1)

        h = 0
        for in_layer, mid_layer in zip(self.in_layers, self.mid_layers):
            h_in = functional.leaky_relu(in_layer(x))
            h = functional.leaky_relu(mid_layer(h + h_in))

        return self.final(h)
        
    def zero_output(self):
        def init(x):
            torch.nn.init.uniform_(x, -1e-5, 1e-5)
        init(self.final.weight)
        init(self.final.bias)


class ChessConvNet(nn.Module):
    def __init__(self, zero=False):
        super().__init__()

        num_filters = 128
        filter_size = 1
        padding = filter_size // 2

        num_hidden = 2048

        self.conv1 = nn.Conv2d(13 + 12, num_filters, filter_size, 1, padding)
        self.conv2 = nn.Conv2d(num_filters, num_filters, filter_size, 1, padding)
        # self.conv3 = nn.Conv2d(d, d, filter_size, 1, padding)
        self.fc1 = nn.Linear(64 * num_filters, num_hidden)

        self.data_linear = nn.Linear(64 * (13 + 12), num_hidden)
        self.data_middle = nn.Linear(num_hidden, num_hidden // 2)
        self.data_final = nn.Linear(num_hidden // 2, 1)

    def forward(self, x0):
        if x0.ndim == 3:
            x0 = x0.view(1, -1, 8, 8)
        output = 0
        x = functional.leaky_relu(self.conv1(x0))
        x = functional.leaky_relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = functional.leaky_relu(self.fc1(x))

        # MLP
        h = self.data_linear(torch.flatten(x0, 1))
        h = functional.leaky_relu(h) + x
        h = functional.leaky_relu(self.data_middle(h))
        h = self.data_final(h)
        output += h
        return output

    def zero_output(self):
        def init(x):
            torch.nn.init.uniform_(x, -1e-5, 1e-5)
        init(self.data_final.weight)
        init(self.data_final.bias)
        init(self.data_linear.weight)
        init(self.data_linear.bias)
        init(self.data_middle.weight)
        init(self.data_middle.bias)
        if hasattr(self, "fc1"):
            init(self.fc1.weight)
            init(self.fc1.bias)
        if hasattr(self, "conv1"):
            init(self.conv1.weight)
            init(self.conv1.bias)
        if hasattr(self, "conv2"):
            init(self.conv2.weight)
            init(self.conv2.bias)


class NetEngine(MinimalEngine):

    def __init__(self, *args, name=None, net=None, from_file=DEFAULT_MODEL_LOCATION):
        super().__init__(*args)
        self.name = name

        blank_board = chess.Board()

        sample_vector = torch.from_numpy(get_features(blank_board).astype(np.float32)).view(1, -1, 8, 8)
        self.net = ChessConvNet()
        # self.net = ChessResnet()

        if from_file:
            self.net.load_state_dict(torch.load(from_file))
        else:
            self.net.eval()
            prev_score = self.net(sample_vector)

            self.net.zero_output()

            new_score = self.net(sample_vector)

            print(f"Set outputs of initial network to have near-zero. "
                  f"Initial board went from {prev_score.data.numpy()[0, 0]} to {new_score.data.numpy()[0, 0]}.")

    def search(self, board: chess.Board, time_limit, ponder, max_tolerance=0.1):
        # returns a random choice among highest-scoring positions
        moves = np.array(list(board.legal_moves))

        features = []
        material_counts = np.zeros(moves.size)
        non_terminal = np.zeros(moves.size)

        for i, move in enumerate(moves):
            board.push(move)
            features.append(get_features(board))
            material_counts[i] = material_count(board)
            if not board.is_game_over():
                non_terminal[i] = 1
            board.pop()

        features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        with torch.no_grad():
            scores = -non_terminal * self.net(features).numpy().ravel() - material_counts

        # consider scores within 0.1 of max to be equivalent
        best_moves = moves[scores.max() - scores <= max_tolerance]

        return np.random.choice(best_moves)


# def plot_filters(model: NetEngine, writer: SummaryWriter, step: int):
#     slices = ["My P", "My N", "My B", "My R", "My Q", "My K", "Op P", "Op N", "Op B", "Op R", "Op Q", "Op K", "Special"]
#
#     if not hasattr(model, "conv1"):
#         return
#
#     weight = model.conv1.weight
#
#     num_filters = weight.shape[0]
#
#     for i in range(len(slices)):
#         grid = make_grid(torch.tile(weight[:, i:(i + 1), :, :], (1, 3, 1, 1)), normalize=True)
#         writer.add_image(slices[i], grid, step)


def expand_state(board):
    """
    Generate a representation of a state for learning that includes:
        - the material count of the state
        - the feature representation of the state
        - the material counts for all child states
        - the feature representations of all child states
    """
    features = get_features(board)
    material = material_count(board)

    next_features = []
    next_material = []
    next_non_terminal = []

    for move in board.legal_moves:
        board.push(move)
        next_features.append(get_features(board))
        next_material.append(material_count(board))
        next_non_terminal.append(0 if board.is_checkmate() else 1)
        board.pop()

    key = board._transposition_key()

    return key, (material, features, next_material, next_features, next_non_terminal)


def update_buffer_bfs(buffer, engine_learner, max_depth=None):
    """
    Fills buffer with BFS.
    """
    if len(buffer) == 0:
        frontier = deque()
        frontier.append((chess.Board(STARTING_POSITION), 0))

        deepest = -1

        known_positions = set()
        known_positions.add(STARTING_POSITION[:-6])

        while frontier:
            board, curr_depth = frontier.popleft()

            key, data = expand_state(board)
            buffer[key] = data

            if max_depth and curr_depth >= max_depth:
                continue

            for move in board.legal_moves:
                new_board = copy.deepcopy(board)
                new_board.push(move)

                new_key = new_board._transposition_key()

                if new_key not in known_positions:
                    frontier.append((new_board, curr_depth + 1))
                    known_positions.add(new_key)

            if curr_depth > deepest:
                print(f"BFS reached depth {curr_depth}. Buffer size {len(buffer)}. Frontier size {len(frontier)}.")
                deepest = curr_depth


def update_buffer_self_play(buffer, engine_learner):
    # play a game against self
    for gn in range(NUM_GAMES):
        board = chess.Board(STARTING_POSITION)
        board.starting_fen = STARTING_POSITION
        while not board.is_game_over(claim_draw=False) and len(board.move_stack) < MAX_MOVES:
            if board._transposition_key() not in buffer:
                key, data = expand_state(board)
                buffer[key] = data
            if random.random() < 0.25:
                move = random.choice(list(board.legal_moves))
            else:
                move = engine_learner.search(board, 10000, 10000, max_tolerance=0.1)
            board.push(move)

            # commented out because there's no need to learn from terminal states
            # if board._transposition_key() not in buffer:
            #     key, data = expand_state(board)
            #     buffer[key] = data

        outcome_str = "mate" if board.is_checkmate() else "draw"
        print(f"Buffer size {len(buffer)}. Game {gn}: {outcome_str}.")


def update_buffer_max_loss(buffer, engine_learner):
    model = engine_learner.net
    for gn in range(NUM_GAMES):
        board = chess.Board(STARTING_POSITION)
        board.starting_fen = STARTING_POSITION
        while not board.is_game_over(claim_draw=False) and len(board.move_stack) < MAX_MOVES:
            if board._transposition_key() not in buffer:
                key, data = expand_state(board)
                buffer[key] = data

            chosen_move = None
            chosen_loss = 0

            moves = np.array(list(board.legal_moves))

            features = []
            material_counts = np.zeros(moves.size)
            non_terminal = np.zeros(moves.size)

            for i, move in enumerate(moves):
                board.push(move)
                features.append(get_features(board))
                material_counts[i] = material_count(board)
                if not board.is_game_over():
                    non_terminal[i] = 1
                board.pop()

            features = torch.from_numpy(np.asarray(features, dtype=np.float32))
            with torch.no_grad():
                scores = -non_terminal * model(features).numpy().ravel() - material_counts

                current_score = material_count(board)
                if not board.is_game_over():
                    current_score += model(torch.from_numpy(get_features(board).astype(np.float32))).numpy().ravel()[0]

            loss = np.abs(current_score - scores)

            best_moves = moves[loss.max() - loss < 1e-3]

            chosen_move = np.random.choice(best_moves)

            board.push(chosen_move)

            # commented out because there's no need to learn from terminal states
            # if board._transposition_key() not in buffer:
            #     key, data = expand_state(board)
            #     buffer[key] = data

        outcome_str = "mate" if board.is_checkmate() else "draw"
        print(f"Buffer size {len(buffer)}. Game {gn}: {outcome_str}.")


def main():
    time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    note = NOTE

    if STORE_LOG:
        base_dir = "/Users/bert/Desktop"
    else:
        base_dir = tempfile.TemporaryDirectory().name

    log_dir = '{}/logs/'.format(base_dir)
    print("Storing logs in {}".format(log_dir))
    writer = SummaryWriter(log_dir + note + time_string, flush_secs=1)

    epoch = 0
    start_time = time.perf_counter()

    if WARM_START:
        engine_learner = NetEngine(None, None, sys.stderr, from_file=DEFAULT_MODEL_LOCATION)
    else:
        engine_learner = NetEngine(None, None, sys.stderr, from_file=None)

    model = engine_learner.net

    opponent_random = RandomMove(None, None, sys.stderr)
    opponent_hanging = HangingEngine(None, None, sys.stderr)
    opponent_minimax = ScoreEngine(None, None, sys.stderr, max_depth=2)

    blank_board = chess.Board()
    sample_vector = get_features(blank_board)
    writer.add_graph(model, torch.Tensor(sample_vector))
    writer.flush()

    game_buffer = deque(maxlen=MAX_BUFFER_ROUNDS)

    total_optimizer_steps = 0

    loss_fn = nn.HuberLoss()
    loss_fn = nn.MSELoss()

    # boards for evaluation
    empty = chess.Board()
    hanging_queen = chess.Board("rn1qkbnr/ppp1pppp/3p4/8/4P1b1/P1N5/1PPP1PPP/R1BQKBNR b KQkq - 0 3")
    ladder_mate = chess.Board("5rr1/1k6/8/8/7K/8/4R3/2R5 w - - 0 1")
    ladder_mate_in_one = chess.Board("5rr1/k7/8/8/7K/8/1R6/2R5 w - - 2 2")
    eval_boards = [empty, hanging_queen, ladder_mate, ladder_mate_in_one]
    eval_board_features = [get_features(x).astype(np.float32) for x in eval_boards]
    eval_board_features = torch.from_numpy(np.asarray(eval_board_features))
    eval_board_material = torch.from_numpy(np.asarray([material_count(x) for x in eval_boards]))
    empty_board_key, empty_board_data = expand_state(empty)
    empty_board_data = list(empty_board_data)
    empty_board_data[2] = []
    empty_board_data[3] = []

    # Anchor boards
    anchor_boards = [empty,]
    anchor_board_values = torch.from_numpy(np.asarray([0,]))
    anchor_board_features = [get_features(x).astype(np.float32) for x in anchor_boards]
    anchor_board_features = torch.from_numpy(np.asarray(anchor_board_features))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # optimizer =  torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 1e-8, 1e-4, step_size_up=100000, cycle_momentum=False,
    # verbose=False)

    eval_time = time.time()
    minimax_eval_time = time.time()

    while True:
        # update engine name
        engine_learner.engine.id["name"] = "NetEngine{}".format(epoch)

        # generate some new positions to put in the buffer

        model.eval()

        if epoch % EPOCHS == 0:
            new_buffer = {}

            # alternate between adding game played against self versus max_play
            if epoch % 2 == 0:
                update_buffer_self_play(new_buffer, engine_learner)
            else:
                update_buffer_max_loss(new_buffer, engine_learner)

            game_buffer.append(new_buffer)

            buffer = {}

            for game in game_buffer:
                buffer.update(game)
            print(f"New games added. Full buffer combining {len(game_buffer)} rounds has {len(buffer)} states")

        if len(game_buffer) < MAX_BUFFER_ROUNDS:
            continue

        # Learning

        loss_record = 0

        if TARGET_UPDATE and epoch % TARGET_UPDATE == 0:
            target_model = copy.deepcopy(model)
            target_model.eval()
            # target_model = model

        if TARGET_UPDATE == 0:
            target_model = model

        batches = batchify(buffer.items(), BATCH_SIZE)

        num_steps = 0
        for batch in batches:
            # train on a batch
            optimizer.zero_grad()
            model.train()

            keys, data_list = zip(*batch)
            keys = list(keys)

            material, features, next_material, next_features, next_non_terminal = zip(*data_list)

            material = torch.from_numpy(np.asarray(material, dtype=np.float32)).view((-1, 1))
            next_material = [torch.from_numpy(np.asarray(m, dtype=np.float32)).view((-1, 1)) for m in next_material]

            features = torch.from_numpy(np.asarray(features, dtype=np.float32))
            next_features = [torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in next_features]
            next_non_terminal = [torch.from_numpy(np.asarray(x, dtype=np.float32)).view((-1, 1))
                                 for x in next_non_terminal]

            current_scores = material + model(features)

            total_loss = ZERO_REGULARIZER * functional.l1_loss(current_scores, material)

            total_loss = ANCHOR_REGULARIZER * functional.l1_loss(model(anchor_board_features).view([-1]),
                                                                 anchor_board_values)

            # collect next features and non-terminal statuses into single batch
            batch_next_features = torch.cat(next_features)

            if batch_next_features.shape[0] <= 1:
                continue

            batch_next_scores = target_model(batch_next_features)

            batch_start_indices = np.cumsum([len(x) for x in next_material])
            batch_start_indices = np.concatenate(([0], batch_start_indices))

            for i in range(len(keys)):
                if len(next_material[i]):
                    next_scores = -next_material[i] - next_non_terminal[i] * \
                                  batch_next_scores[batch_start_indices[i]:(batch_start_indices[i + 1])]
                    next_score = torch.max(next_scores)

                    # best_move = torch.argmax(next_scores)
                    # next_score = -next_material[i][best_move] - next_non_terminal[i][best_move] * target_model(
                    #     batch_next_features[batch_start_indices[i] + best_move])

                    # assert(torch.allclose(next_score, next_score_old))

                    loss = loss_fn(current_scores[i], next_score.view([1]))
                    total_loss += loss

                    margin_loss = functional.margin_ranking_loss(current_scores[i], next_scores, torch.ones((1, 1)))

                    total_loss += margin_loss

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            # scheduler.step()

            loss_record += total_loss.data

            num_steps += 1
            total_optimizer_steps += 1

            writer.add_scalar("Optimizer loss", total_loss.data / len(keys), total_optimizer_steps)

        model.eval()

        writer.add_scalar("Learning Loss", loss_record / len(buffer), epoch)
        # scheduler.step(loss_record)
        # scheduler.step()

        # log diagnostic info
        with torch.no_grad():
            eval_board_scores = model(eval_board_features).ravel() + eval_board_material
            writer.add_scalar("Empty Board Score", eval_board_scores[0], epoch * 1000)
            writer.add_scalar("Hanging Queen Score", eval_board_scores[1].data, epoch)
            writer.add_scalar("Ladder Mate Score", eval_board_scores[2].data, epoch)
            writer.add_scalar("Ladder Mate-in-1 Score", eval_board_scores[3].data, epoch)

            # eval_board_scores = target_model(eval_board_features).ravel() + eval_board_material
            # writer.add_scalar("Target Empty Board Score", eval_board_scores[0], epoch)
            # writer.add_scalar("Target Hanging Queen Score", eval_board_scores[1].data, step)
            # writer.add_scalar("Target Ladder Mate Score", eval_board_scores[2].data, step)

        if time.time() - eval_time > EVAL_INTERVAL:
            print("Evaluating")
            eval_time = time.time()
            num_games = 10
            rand_wins, rand_losses, rand_ties, rand_score = play_match(engine_learner, opponent_random, num_games,
                                                                       writer, "Random",
                                                                       epoch,
                                                                       starting_position=STARTING_POSITION)

            hang_wins, hang_losses, hang_ties, hang_score = play_match(engine_learner, opponent_hanging, num_games,
                                                                       writer, "Hanging",
                                                                       epoch,
                                                                       starting_position=STARTING_POSITION)

            play_match(engine_learner, engine_learner, num_games, writer, "Self", epoch,
                       starting_position=STARTING_POSITION)

            writer.add_scalar("Win Rate v. Random", rand_wins / num_games, epoch)
            writer.add_scalar("Loss Rate v. Random", rand_losses / num_games, epoch)
            writer.add_scalar("Win Rate v. Hanging", hang_wins / num_games, epoch)
            writer.add_scalar("Loss Rate v. Hanging", hang_losses / num_games, epoch)
            writer.add_scalar("Score v. Hanging", hang_score, epoch)
            writer.add_scalar("Score v. Random", rand_score, epoch)
            if STORE_LOG:
                torch.save(model.state_dict(), DEFAULT_MODEL_LOCATION)

            if time.time() - minimax_eval_time > EVAL_INTERVAL * 10:
                # compare against slow minimax agent

                mini_wins, mini_losses, mini_ties, mini_score = play_match(engine_learner, opponent_minimax, num_games,
                                                                           writer, "Minimax", epoch,
                                                                           starting_position=STARTING_POSITION)

                minimax_eval_time = time.time()

                writer.add_scalar("Win Rate v. Minimax", mini_wins / num_games, epoch)
                writer.add_scalar("Loss Rate v. Minimax", mini_losses / num_games, epoch)
                writer.add_scalar("Score v. Minimax", mini_score, epoch)

            # plot_filters(model, writer, total_optimizer_steps)

        writer.add_scalar("Buffer Size", len(buffer), epoch)

        elapsed_time = time.perf_counter() - start_time

        epoch += 1
        
        print("Completed {} epochs ({:.2f} rounds/sec). Loss {}".format(epoch, epoch / elapsed_time,
                                                                        loss_record / len(buffer)))
        writer.flush()

        epoch_loss = loss_record / len(buffer)


if __name__ == "__main__":
    main()
