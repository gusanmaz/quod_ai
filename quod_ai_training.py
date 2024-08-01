import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from datetime import timedelta
from typing import List, Tuple, Optional
import argparse
import os

from quod_game import QuodGame, BOARD_SIZE, PLAYER_RED, PLAYER_BLUE, QUOD_RED, QUOD_BLUE, QUAZAR, EMPTY_CELL

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and configured for use.")
else:
    print("No GPU found. Running on CPU.")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def create_model():
    with tf.device('/GPU:0'):
        board_input = keras.layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, 4))  # 4 channels: red, blue, quazar, empty
        hardness_input = keras.layers.Input(shape=(1,))

        x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(board_input)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Flatten()(x)

        x = keras.layers.Concatenate()([x, hardness_input])

        x = keras.layers.Dense(256, activation='relu')(x)
        output = keras.layers.Dense(BOARD_SIZE * BOARD_SIZE, activation='softmax')(x)

        model = keras.Model(inputs=[board_input, hardness_input], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model

class QuodAI:
    def __init__(self, hardness: float, model: Optional[keras.Model] = None):
        self.hardness = hardness
        self.model = model

    @tf.function
    def predict(self, state, hardness):
        return self.model([state, hardness], training=False)

    def get_move(self, game: QuodGame, use_strategy: bool = True) -> Tuple[Optional[Tuple[int, int]], Optional[str]]:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return None, None

        if use_strategy and np.random.random() < self.hardness:
            # Try strategic moves first
            strategic_move = self.get_strategic_move(game)
            if strategic_move:
                return strategic_move, 'quod'

        # Neural network strategy
        if self.model:
            state = self.board_to_channels(game.board)
            state = tf.convert_to_tensor(state.reshape(1, BOARD_SIZE, BOARD_SIZE, 4), dtype=tf.float32)
            hardness = tf.convert_to_tensor([[self.hardness]], dtype=tf.float32)
            move_probs = self.predict(state, hardness)[0].numpy()

            # Apply mask for legal moves
            mask = np.zeros(BOARD_SIZE * BOARD_SIZE)
            for move in legal_moves:
                mask[move[0] * BOARD_SIZE + move[1]] = 1
            move_probs *= mask
            move_probs_sum = np.sum(move_probs)
            if move_probs_sum > 0:
                move_probs /= move_probs_sum
            else:
                move_probs = mask / np.sum(mask)

            move = np.unravel_index(np.argmax(move_probs), (BOARD_SIZE, BOARD_SIZE))
            return move, 'quod' if game.quods[game.current_player] > 0 else 'quazar'

        # Random move with preference for quods
        piece_type = 'quod' if game.quods[game.current_player] > 0 else 'quazar'
        return legal_moves[np.random.randint(len(legal_moves))], piece_type

    def board_to_channels(self, board):
        channels = np.zeros((BOARD_SIZE, BOARD_SIZE, 4), dtype=np.float32)
        channels[:, :, 0] = (board == QUOD_RED).astype(np.float32)
        channels[:, :, 1] = (board == QUOD_BLUE).astype(np.float32)
        channels[:, :, 2] = (board == QUAZAR).astype(np.float32)
        channels[:, :, 3] = (board == EMPTY_CELL).astype(np.float32)
        return channels

    def get_strategic_move(self, game: QuodGame) -> Optional[Tuple[int, int]]:
        # Check for winning move
        winning_move = self.find_winning_move(game)
        if winning_move:
            return winning_move

        # Check for opponent's winning move and block it
        opponent_winning_move = self.find_winning_move(game, opponent=True)
        if opponent_winning_move:
            return opponent_winning_move

        # Try to form third corner of a square
        third_corner_move = self.find_third_corner_move(game)
        if third_corner_move:
            return third_corner_move

        # Play in the center if it's available
        center = BOARD_SIZE // 2
        if game.board[center, center] == EMPTY_CELL:
            return (center, center)

        return None

    def find_winning_move(self, game: QuodGame, opponent: bool = False) -> Optional[Tuple[int, int]]:
        player = PLAYER_BLUE if opponent else game.current_player
        player_quod = QUOD_BLUE if player == PLAYER_BLUE else QUOD_RED
        for move in game.get_legal_moves():
            game.board[move[0], move[1]] = player_quod
            if game.check_for_square(player):
                game.board[move[0], move[1]] = EMPTY_CELL
                return move
            game.board[move[0], move[1]] = EMPTY_CELL
        return None

    def find_third_corner_move(self, game: QuodGame) -> Optional[Tuple[int, int]]:
        player_quod = QUOD_RED if game.current_player == PLAYER_RED else QUOD_BLUE
        player_pegs = np.argwhere(game.board == player_quod)
        for i in range(len(player_pegs)):
            for j in range(i + 1, len(player_pegs)):
                p1, p2 = player_pegs[i], player_pegs[j]
                potential_corners = [
                    (p1[0], p2[1]),
                    (p2[0], p1[1])
                ]
                for corner in potential_corners:
                    if corner in game.get_legal_moves():
                        opposite_corner = (p1[0] + p2[0] - corner[0], p1[1] + p2[1] - corner[1])
                        if opposite_corner in game.get_legal_moves():
                            return corner
        return None

def play_game(ai_red: QuodAI, ai_blue: QuodAI, use_strategy: bool = True) -> Tuple[
    int, List[Tuple[np.ndarray, float, Tuple[int, int]]]]:
    game = QuodGame()
    moves = []
    while not game.is_terminal():
        ai = ai_red if game.current_player == PLAYER_RED else ai_blue
        move, piece_type = ai.get_move(game, use_strategy)

        if move is None:
            break

        moves.append((game.board.copy(), ai.hardness, move))
        game.make_move(move[0], move[1], piece_type)

    return game.get_winner(), moves

def train_model(model: keras.Model, num_games: int = 100000):
    start_time = time.time()
    games_per_update = 100

    ai_strong = QuodAI(hardness=0.8, model=model)
    ai_weak = QuodAI(hardness=0.2, model=model)

    print("Starting training process...")
    print(f"Total games to play: {num_games}")

    total_moves = 0

    for game_num in range(num_games):
        if game_num == 0:
            print("Playing first game...")

        winner, moves = play_game(ai_strong, ai_weak)
        total_moves += len(moves)

        # Train on this game
        if moves:
            states = [ai_strong.board_to_channels(move[0]) for move in moves]
            states = np.array(states)
            hardness_values = np.array([move[1] for move in moves])
            actions = np.zeros((len(moves), BOARD_SIZE * BOARD_SIZE))
            for i, move in enumerate(moves):
                actions[i, move[2][0] * BOARD_SIZE + move[2][1]] = 1

            rewards = np.array([winner * (0.9 ** i) for i in range(len(moves))][::-1])

            if game_num == 0:
                print("Training on first game...")

            model.fit([states, hardness_values], actions, sample_weight=rewards, epochs=1, verbose=0)

            if game_num == 0:
                print("First game completed and trained on.")

        if (game_num + 1) % games_per_update == 0 or game_num == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            games_per_second = (game_num + 1) / elapsed_time
            estimated_total_time = num_games / games_per_second
            estimated_remaining_time = estimated_total_time - elapsed_time

            print(f"\nProgress: {game_num + 1}/{num_games} games ({(game_num + 1) / num_games * 100:.2f}%)")
            print(f"Elapsed time: {timedelta(seconds=int(elapsed_time))}")
            print(f"Estimated remaining time: {timedelta(seconds=int(estimated_remaining_time))}")
            print(f"Games per second: {games_per_second:.2f}")
            print(f"Average game length: {total_moves / (game_num + 1):.2f} moves")
            print("---")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {timedelta(seconds=int(total_time))}")
    print(f"Total games played: {num_games}")
    print(f"Average game length: {total_moves / num_games:.2f} moves")
    print(f"Average games per second: {num_games / total_time:.2f}")
    return model

def board_to_markdown(board):
    symbols = {EMPTY_CELL: ' ', QUOD_RED: 'R', QUOD_BLUE: 'B', QUAZAR: 'Q'}
    markdown = "| | " + " | ".join(str(i) for i in range(BOARD_SIZE)) + " |\n"
    markdown += "|" + "-|" * (BOARD_SIZE + 1) + "\n"
    for i, row in enumerate(board):
        markdown += f"|{i}| " + " | ".join(symbols.get(cell, 'X') for cell in row) + " |\n"
    return markdown

def evaluate_ai(model: keras.Model, games: int = 1000, use_strategy: bool = True) -> dict:
    ai_strong = QuodAI(hardness=0.9, model=model)
    ai_medium = QuodAI(hardness=0.6, model=model)
    ai_weak = QuodAI(hardness=0.3, model=model)

    results = {
        "Strong vs Medium": [0, 0, 0],  # [Strong wins, Medium wins, Draws]
        "Strong vs Weak": [0, 0, 0],    # [Strong wins, Weak wins, Draws]
        "Medium vs Weak": [0, 0, 0]     # [Medium wins, Weak wins, Draws]
    }

    strategy_type = "Combined Strategy" if use_strategy else "NN Only"
    log_filename = f"evaluation_{strategy_type.lower().replace(' ', '_')}.md"

    print(f"Starting evaluation with {games} games for each matchup...")
    print(f"Using {strategy_type} for decision making.")
    print(f"Logging games to {log_filename}")

    start_time = time.time()

    with open(log_filename, 'w') as log_file:
        for matchup in results.keys():
            log_file.write(f"# {matchup} {strategy_type}\n\n")
            for i in range(games):
                if i % 100 == 0:
                    elapsed_time = time.time() - start_time
                    games_per_second = (i * 3) / elapsed_time  # 3 matchups
                    print(f"Completed {i} games for matchup {matchup}...")
                    print(f"Games per second: {games_per_second:.2f}")

                ai1, ai2 = (ai_strong, ai_medium) if matchup == "Strong vs Medium" else \
                    (ai_strong, ai_weak) if matchup == "Strong vs Weak" else \
                        (ai_medium, ai_weak)

                winner, moves = play_game(ai1, ai2, use_strategy)
                if winner == PLAYER_RED:
                    results[matchup][0] += 1
                elif winner == PLAYER_BLUE:
                    results[matchup][1] += 1
                else:
                    results[matchup][2] += 1

                log_file.write(f"## Game {i + 1}\n\n")
                for turn, (board, _, move) in enumerate(moves):
                    log_file.write(f"### Turn {turn + 1}\n\n")
                    log_file.write(board_to_markdown(board) + "\n\n")

                log_file.write(f"**Result: {'AI 1' if winner == PLAYER_RED else 'AI 2' if winner == PLAYER_BLUE else 'Draw'}**\n\n")

                # Play reverse game
                winner, moves = play_game(ai2, ai1, use_strategy)
                if winner == PLAYER_RED:
                    results[matchup][1] += 1
                elif winner == PLAYER_BLUE:
                    results[matchup][0] += 1
                else:
                    results[matchup][2] += 1

                log_file.write(f"## Game {i + 1} (Reverse)\n\n")
                for turn, (board, _, move) in enumerate(moves):
                    log_file.write(f"### Turn {turn + 1}\n\n")
                    log_file.write(board_to_markdown(board) + "\n\n")

                log_file.write(
                    f"**Result: {'AI 2' if winner == PLAYER_RED else 'AI 1' if winner == PLAYER_BLUE else 'Draw'}**\n\n")

                # Write current results after each pair of games
                log_file.write(
                    f"### Current Results: {results[matchup][0]} - {results[matchup][1]} - {results[matchup][2]}\n\n")
                log_file.flush()  # Ensure data is written to file

                # Write final results for this matchup
            log_file.write(f"# Final Results for {matchup}\n\n")
            log_file.write(f"**{results[matchup][0]} - {results[matchup][1]} - {results[matchup][2]}**\n\n")

            # Print results to console
            print(f"{matchup}: {results[matchup][0]} - {results[matchup][1]} - {results[matchup][2]}")

        total_time = time.time() - start_time
        print(f"\nEvaluation completed in {timedelta(seconds=int(total_time))}")
        print(f"Average games per second: {(games * 3 * 2) / total_time:.2f}")

        print("\nFinal Results:")
        for matchup, scores in results.items():
            print(f"{matchup}: {scores[0]} - {scores[1]} - {scores[2]}")

        return results

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Quod AI Training and Evaluation")
        parser.add_argument("--mode", choices=["train", "evaluate"], required=True, help="Mode of operation")
        parser.add_argument("--games", type=int, default=10000, help="Number of games for training or evaluation")
        args = parser.parse_args()

        if args.mode == "train":
            model = create_model()
            trained_model = train_model(model, num_games=args.games)
            trained_model.save('quod_ai_model.h5')
            print("Model trained and saved as 'quod_ai_model.h5'")
        elif args.mode == "evaluate":
            print("Loading the saved model...")
            trained_model = keras.models.load_model('quod_ai_model.h5')
            print("Model loaded successfully.")

            print("\nEvaluating with combined strategy and NN...")
            results_with_strategy = evaluate_ai(trained_model, games=args.games, use_strategy=True)

            print("\nEvaluating with only NN...")
            results_only_nn = evaluate_ai(trained_model, games=args.games, use_strategy=False)

            print("\nComparison of results:")
            for matchup in results_with_strategy.keys():
                print(f"\n{matchup}:")
                print(
                    f"  With strategy: {results_with_strategy[matchup][0]} - {results_with_strategy[matchup][1]} - {results_with_strategy[matchup][2]}")
                print(
                    f"  Only NN:      {results_only_nn[matchup][0]} - {results_only_nn[matchup][1]} - {results_only_nn[matchup][2]}")