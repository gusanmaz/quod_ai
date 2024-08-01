# Project Structure

```
.
├── .idea
├── .idea
│   ├── .gitignore
├── .idea
│   ├── workspace.xml
├── __pycache__
├── quod_ai_training.py
├── quod_game.py
├── quod_visualisation_canvas.py
├── quod_visualisation_svg.py
├── quod_visualization.py
├── random_games
├── random_games_100
├── random_games_not_bad
└── random_play.py
```

## Shell Commands to Create Project Structure

```bash
mkdir -p ".idea"
mkdir -p "__pycache__"
mkdir -p "random_games"
mkdir -p "random_games_100"
mkdir -p "random_games_not_bad"
mkdir -p ".idea"
touch ".idea/.gitignore"
mkdir -p ".idea"
touch ".idea/workspace.xml"
touch "quod_ai_training.py"
touch "quod_game.py"
touch "quod_visualisation_canvas.py"
touch "quod_visualisation_svg.py"
touch "quod_visualization.py"
touch "random_play.py"
```

## File Contents

### .idea/.gitignore

```Ignore List
# Default ignored files
/shelf/
/workspace.xml
# Editor-based HTTP Client requests
/httpRequests/
# Datasource local storage ignored files
/dataSources/
/dataSources.local.xml

```

### .idea/workspace.xml

```XML
<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectViewState">
    <option name="hideEmptyMiddlePackages" value="true" />
    <option name="showLibraryContents" value="true" />
  </component>
</project>
```

### quod_ai_training.py

```Python
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
```

### quod_game.py

```Python
import numpy as np
from typing import List, Tuple
from itertools import combinations

# Constants
BOARD_SIZE = 11
INITIAL_QUODS = 20
INITIAL_QUAZARS = 6

# Cell states
EMPTY_CELL = 0
QUOD_RED = 1
QUOD_BLUE = 2
QUAZAR = 3
DISABLED_CELL = -1

# Players
PLAYER_RED = 1
PLAYER_BLUE = 2


class QuodGame:
    def __init__(self):
        self.board = np.full((BOARD_SIZE, BOARD_SIZE), EMPTY_CELL, dtype=np.int8)
        self.quazar_owners = np.full((BOARD_SIZE, BOARD_SIZE), 0, dtype=np.int8)
        self.current_player = PLAYER_RED
        self.quods = {PLAYER_RED: INITIAL_QUODS, PLAYER_BLUE: INITIAL_QUODS}
        self.quazars = {PLAYER_RED: INITIAL_QUAZARS, PLAYER_BLUE: INITIAL_QUAZARS}
        self.remove_corners()
        self.moves = []
        self.current_turn = 1
        self.current_turn_moves = []
        self.last_move = None
        self.winning_square = None

    def remove_corners(self):
        self.board[0, 0] = self.board[0, -1] = self.board[-1, 0] = self.board[-1, -1] = DISABLED_CELL

    def make_move(self, row: int, col: int, piece_type: int) -> bool:
        if self.board[row, col] != EMPTY_CELL:
            return False
        if self.quods[self.current_player] == 0 and piece_type in [QUOD_RED, QUOD_BLUE]:
            return False
        if (self.current_player == PLAYER_RED and piece_type == QUOD_BLUE) or \
                (self.current_player == PLAYER_BLUE and piece_type == QUOD_RED):
            return False

        self.board[row, col] = piece_type
        self.last_move = (row, col)

        if piece_type in [QUOD_RED, QUOD_BLUE]:
            self.quods[self.current_player] -= 1
            self.current_turn_moves.append((self.current_player, piece_type, (row, col)))

            # Check for square formation immediately after placing a quod
            self.winning_square = self.check_for_square(self.current_player)
            if self.winning_square:
                self.end_turn()
                return True

            self.end_turn()
        elif piece_type == QUAZAR:
            if self.quazars[self.current_player] > 0:
                self.quazars[self.current_player] -= 1
                self.quazar_owners[row, col] = self.current_player
                self.current_turn_moves.append((self.current_player, piece_type, (row, col)))
            else:
                return False
        else:
            return False

        return True

    def end_turn(self):
        self.moves.append((self.current_turn, self.current_player, self.current_turn_moves.copy()))
        self.current_player = PLAYER_BLUE if self.current_player == PLAYER_RED else PLAYER_RED
        self.current_turn_moves = []
        if self.current_player == PLAYER_RED:
            self.current_turn += 1

    def get_legal_moves(self) -> List[Tuple[int, int]]:
        return [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if self.board[r, c] == EMPTY_CELL]

    def is_terminal(self) -> bool:
        return self.winning_square is not None or (self.quods[PLAYER_RED] == 0 and self.quods[PLAYER_BLUE] == 0)

    def check_for_square(self, player: int) -> List[Tuple[Tuple[int, int], ...]]:
        player_quod = QUOD_RED if player == PLAYER_RED else QUOD_BLUE
        player_pegs = np.argwhere(self.board == player_quod)
        squares = []

        if self.last_move is not None and len(player_pegs) >= 4:
            other_pegs = [tuple(peg) for peg in player_pegs if tuple(peg) != self.last_move]
            for three_pegs in combinations(other_pegs, 3):
                four_pegs = (self.last_move,) + three_pegs
                if self.is_square(four_pegs):
                    squares.append(four_pegs)

        return squares[0] if squares else None

    def is_square(self, points: List[Tuple[int, int]]) -> bool:
        def distance_sq(point1, point2):
            return (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2

        distances = [distance_sq(p1, p2) for p1, p2 in combinations(points, 2)]

        sorted_distances = sorted(set(distances))

        if len(sorted_distances) != 2:
            return False

        side_sq, diagonal_sq = sorted_distances

        rel_tol = 1e-9

        if not np.isclose(diagonal_sq, 2 * side_sq, rtol=rel_tol):
            return False

        side_count = sum(1 for d in distances if np.isclose(d, side_sq, rtol=rel_tol))
        diagonal_count = sum(1 for d in distances if np.isclose(d, diagonal_sq, rtol=rel_tol))

        return side_count == 4 and diagonal_count == 2

    def get_winner(self) -> int:
        if self.winning_square:
            return PLAYER_RED if self.board[
                                     self.winning_square[0][0], self.winning_square[0][1]] == QUOD_RED else PLAYER_BLUE
        elif self.quods[PLAYER_RED] == 0 and self.quods[PLAYER_BLUE] == 0:
            if self.quazars[PLAYER_RED] > self.quazars[PLAYER_BLUE]:
                return PLAYER_RED
            elif self.quazars[PLAYER_BLUE] > self.quazars[PLAYER_RED]:
                return PLAYER_BLUE
        return 0  # Draw or game not finished

    def __str__(self):
        symbols = {EMPTY_CELL: ' ', QUOD_RED: 'R', QUOD_BLUE: 'B', QUAZAR: 'Q', DISABLED_CELL: 'X'}
        return '\n'.join([''.join([symbols[cell] for cell in row]) for row in self.board])
```

### quod_visualisation_canvas.py

```Python
import html
from quod_game import QUOD_RED, QUOD_BLUE, QUAZAR, EMPTY_CELL, DISABLED_CELL, BOARD_SIZE, PLAYER_RED, PLAYER_BLUE


def board_to_canvas(board, quazar_owners, last_move=None, winning_square=None):
    cell_size = 30
    board_size = cell_size * BOARD_SIZE

    canvas_script = f"""
    <canvas id="quodBoard" width="{board_size}" height="{board_size}"></canvas>
    <script>
        const canvas = document.getElementById('quodBoard');
        const ctx = canvas.getContext('2d');
        const cellSize = {cell_size};

        // Draw grid
        ctx.strokeStyle = 'black';
        for (let i = 0; i <= {BOARD_SIZE}; i++) {{
            ctx.beginPath();
            ctx.moveTo(0, i * cellSize);
            ctx.lineTo({board_size}, i * cellSize);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(i * cellSize, 0);
            ctx.lineTo(i * cellSize, {board_size});
            ctx.stroke();
        }}

        // Draw cells
        const board = {board};
        const quazar_owners = {quazar_owners};
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = '20px Arial';
        for (let i = 0; i < {BOARD_SIZE}; i++) {{
            for (let j = 0; j < {BOARD_SIZE}; j++) {{
                const x = j * cellSize;
                const y = i * cellSize;
                if (board[i][j] === {QUOD_RED}) {{
                    ctx.fillStyle = 'red';
                    ctx.fillRect(x, y, cellSize, cellSize);
                    ctx.fillStyle = 'white';
                    ctx.fillText('R', x + cellSize/2, y + cellSize/2);
                }} else if (board[i][j] === {QUOD_BLUE}) {{
                    ctx.fillStyle = 'blue';
                    ctx.fillRect(x, y, cellSize, cellSize);
                    ctx.fillStyle = 'white';
                    ctx.fillText('B', x + cellSize/2, y + cellSize/2);
                }} else if (board[i][j] === {QUAZAR}) {{
                    ctx.fillStyle = quazar_owners[i][j] === {PLAYER_RED} ? 'red' : 'blue';
                    ctx.fillText('Q', x + cellSize/2, y + cellSize/2);
                }} else if (board[i][j] === {DISABLED_CELL}) {{
                    ctx.fillStyle = 'gray';
                    ctx.fillRect(x, y, cellSize, cellSize);
                }}
            }}
        }}

        // Highlight last move
        if ({last_move}) {{
            const [row, col] = {last_move};
            ctx.strokeStyle = 'green';
            ctx.lineWidth = 3;
            ctx.strokeRect(col * cellSize, row * cellSize, cellSize, cellSize);
        }}

        // Draw winning square
        if ({winning_square}) {{
            const square = {winning_square};
            ctx.strokeStyle = 'yellow';
            ctx.lineWidth = 3;
            for (let i = 0; i < 4; i++) {{
                const [r1, c1] = square[i];
                const [r2, c2] = square[(i + 1) % 4];
                ctx.beginPath();
                ctx.moveTo(c1 * cellSize + cellSize/2, r1 * cellSize + cellSize/2);
                ctx.lineTo(c2 * cellSize + cellSize/2, r2 * cellSize + cellSize/2);
                ctx.stroke();
            }}
        }}
    </script>
    """
    return canvas_script

def save_game_to_html(game, filename, player1, player2):
    with open(filename, 'w') as f:
        f.write("<html><head><title>Quod Game</title></head><body>")
        f.write(
            f"<h1>Quod Game: <span style='color: red;'>{html.escape(player1)}</span> vs <span style='color: blue;'>{html.escape(player2)}</span></h1>")

        board = [[EMPTY_CELL for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        quazar_owners = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[0][0] = board[0][-1] = board[-1][0] = board[-1][-1] = DISABLED_CELL

        for turn, player, moves in game.moves:
            f.write(f"<h2>Turn {turn}</h2>")
            f.write(
                f"<p>Player: <span style='color: {'red' if player == PLAYER_RED else 'blue'};'>{'Red' if player == PLAYER_RED else 'Blue'}</span></p>")

            for idx, (_, piece_type, move) in enumerate(moves):
                row, col = move
                board[row][col] = piece_type
                if piece_type == QUAZAR:
                    quazar_owners[row][col] = player

                f.write(f"<h3>Move {idx + 1}</h3>")
                f.write(
                    f"<p>Piece: {'Quod' if piece_type in [QUOD_RED, QUOD_BLUE] else 'Quazar'}, Position: {move}</p>")

                f.write(board_to_canvas(board, quazar_owners, move, game.winning_square))

        winner = game.get_winner()
        winner_str = "Red" if winner == PLAYER_RED else "Blue" if winner == PLAYER_BLUE else "Draw"
        f.write(
            f"<h2>Final Result: <span style='color: {'red' if winner == PLAYER_RED else 'blue' if winner == PLAYER_BLUE else 'black'};'>{winner_str}</span></h2>")

        f.write("</body></html>")
```

### quod_visualisation_svg.py

```Python
import html
from quod_game import QUOD_RED, QUOD_BLUE, QUAZAR, EMPTY_CELL, DISABLED_CELL, BOARD_SIZE, PLAYER_RED, PLAYER_BLUE


def board_to_svg(board, quazar_owners, last_move=None, winning_square=None):
    cell_size = 30
    board_size = cell_size * BOARD_SIZE

    svg_content = f'<svg width="{board_size}" height="{board_size}" xmlns="http://www.w3.org/2000/svg">'

    # Draw the grid
    for i in range(BOARD_SIZE + 1):
        svg_content += f'<line x1="0" y1="{i * cell_size}" x2="{board_size}" y2="{i * cell_size}" stroke="black" />'
        svg_content += f'<line x1="{i * cell_size}" y1="0" x2="{i * cell_size}" y2="{board_size}" stroke="black" />'

    # Draw the cells
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            x = j * cell_size
            y = i * cell_size

            if board[i][j] == QUOD_RED:
                svg_content += f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="red" />'
                svg_content += f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2}" fill="white" text-anchor="middle" dominant-baseline="middle">R</text>'
            elif board[i][j] == QUOD_BLUE:
                svg_content += f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="blue" />'
                svg_content += f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2}" fill="white" text-anchor="middle" dominant-baseline="middle">B</text>'
            elif board[i][j] == QUAZAR:
                quazar_color = "red" if quazar_owners[i][j] == PLAYER_RED else "blue"
                svg_content += f'<text x="{x + cell_size / 2}" y="{y + cell_size / 2}" fill="{quazar_color}" text-anchor="middle" dominant-baseline="middle">Q</text>'
            elif board[i][j] == DISABLED_CELL:
                svg_content += f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="gray" />'

    # Highlight last move
    if last_move:
        x = last_move[1] * cell_size
        y = last_move[0] * cell_size
        svg_content += f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="none" stroke="green" stroke-width="3" />'

    # Draw winning square
    if winning_square:
        svg_content += '<g stroke="yellow" stroke-width="3" fill="none">'
        for i in range(4):
            x1, y1 = winning_square[i]
            x2, y2 = winning_square[(i + 1) % 4]
            svg_content += f'<line x1="{x1 * cell_size + cell_size / 2}" y1="{y1 * cell_size + cell_size / 2}" x2="{x2 * cell_size + cell_size / 2}" y2="{y2 * cell_size + cell_size / 2}" />'
        svg_content += '</g>'

    svg_content += '</svg>'
    return svg_content


def save_game_to_html(game, filename, player1, player2):
    with open(filename, 'w') as f:
        f.write("<html><head><title>Quod Game</title></head><body>")
        f.write(
            f"<h1>Quod Game: <span style='color: red;'>{html.escape(player1)}</span> vs <span style='color: blue;'>{html.escape(player2)}</span></h1>")

        board = [[EMPTY_CELL for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        quazar_owners = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[0][0] = board[0][-1] = board[-1][0] = board[-1][-1] = DISABLED_CELL

        for turn, player, moves in game.moves:
            f.write(f"<h2>Turn {turn}</h2>")
            f.write(
                f"<p>Player: <span style='color: {'red' if player == PLAYER_RED else 'blue'};'>{'Red' if player == PLAYER_RED else 'Blue'}</span></p>")

            for idx, (_, piece_type, move) in enumerate(moves):
                row, col = move
                board[row][col] = piece_type
                if piece_type == QUAZAR:
                    quazar_owners[row][col] = player

                f.write(f"<h3>Move {idx + 1}</h3>")
                f.write(
                    f"<p>Piece: {'Quod' if piece_type in [QUOD_RED, QUOD_BLUE] else 'Quazar'}, Position: {move}</p>")

                f.write(board_to_svg(board, quazar_owners, move, game.winning_square))

        winner = game.get_winner()
        winner_str = "Red" if winner == PLAYER_RED else "Blue" if winner == PLAYER_BLUE else "Draw"
        f.write(
            f"<h2>Final Result: <span style='color: {'red' if winner == PLAYER_RED else 'blue' if winner == PLAYER_BLUE else 'black'};'>{winner_str}</span></h2>")

        f.write("</body></html>")



```

### quod_visualization.py

```Python
import html
from quod_game import QUOD_RED, QUOD_BLUE, QUAZAR, EMPTY_CELL, DISABLED_CELL, BOARD_SIZE, PLAYER_RED, PLAYER_BLUE


def board_to_html(board, last_move=None, winning_square=None):
    cell_size = 30

    html_content = f"""
    <html>
    <head>
        <style>
            table {{ border-collapse: collapse; }}
            td {{
                width: {cell_size}px;
                height: {cell_size}px;
                border: 1px solid black;
                text-align: center;
                font-weight: bold;
            }}
            .red {{ background-color: red; color: white; }}
            .blue {{ background-color: blue; color: white; }}
            .quazar-red {{ color: red; }}
            .quazar-blue {{ color: blue; }}
            .last-move {{ border: 3px solid green; }}
            .disabled {{ background-color: gray; }}
            .winning-square {{ border: 3px solid yellow; }}
        </style>
    </head>
    <body>
        <table>
    """

    for i in range(BOARD_SIZE):
        html_content += "<tr>"
        for j in range(BOARD_SIZE):
            cell_class = ""
            cell_content = "&nbsp;"

            if board[i][j] == QUOD_RED:
                cell_class = "red"
                cell_content = "R"
            elif board[i][j] == QUOD_BLUE:
                cell_class = "blue"
                cell_content = "B"
            elif board[i][j] == QUAZAR:
                cell_class = "quazar-red" if board[i][j] == PLAYER_RED else "quazar-blue"
                cell_content = "Q"
            elif board[i][j] == DISABLED_CELL:
                cell_class = "disabled"

            if last_move and (i, j) == last_move:
                cell_class += " last-move"

            if winning_square and (i, j) in winning_square:
                cell_class += " winning-square"

            html_content += f'<td class="{cell_class}">{cell_content}</td>'

        html_content += "</tr>"

    html_content += """
        </table>
    </body>
    </html>
    """

    return html_content

def save_game_to_html(game, filename, player1, player2):
    with open(filename, 'w') as f:
        f.write("<html><head><title>Quod Game</title></head><body>")
        f.write(
            f"<h1>Quod Game: <span style='color: red;'>{html.escape(player1)}</span> vs <span style='color: blue;'>{html.escape(player2)}</span></h1>")

        board = [[EMPTY_CELL for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        board[0][0] = board[0][-1] = board[-1][0] = board[-1][-1] = DISABLED_CELL

        for turn, player, moves in game.moves:
            f.write(f"<h2>Turn {turn}</h2>")
            f.write(
                f"<p>Player: <span style='color: {'red' if player == PLAYER_RED else 'blue'};'>{'Red' if player == PLAYER_RED else 'Blue'}</span></p>")

            for idx, (_, piece_type, move) in enumerate(moves):
                row, col = move
                board[row][col] = piece_type

                f.write(f"<h3>Move {idx + 1}</h3>")
                f.write(
                    f"<p>Piece: {'Quod' if piece_type in [QUOD_RED, QUOD_BLUE] else 'Quazar'}, Position: {move}</p>")

                f.write(board_to_html(board, move, game.winning_square))

        winner = game.get_winner()
        winner_str = "Red" if winner == PLAYER_RED else "Blue" if winner == PLAYER_BLUE else "Draw"
        f.write(
            f"<h2>Final Result: <span style='color: {'red' if winner == PLAYER_RED else 'blue' if winner == PLAYER_BLUE else 'black'};'>{winner_str}</span></h2>")

        f.write("</body></html>")
```

### random_play.py

```Python
import os
import argparse
from quod_game import QuodGame, PLAYER_RED, PLAYER_BLUE, QUOD_RED, QUOD_BLUE, QUAZAR
import quod_visualisation_svg
import quod_visualisation_canvas
import quod_visualization
import random

can = quod_visualisation_canvas.save_game_to_html
tab = quod_visualization.save_game_to_html
svg = quod_visualisation_svg.save_game_to_html

def import_visualization(vis_type):
    if vis_type == 'table':
        save_game_to_html = tab
    elif vis_type == 'svg':
        save_game_to_html = svg
    elif vis_type == 'canvas':
        save_game_to_html = can
    else:
        raise ValueError(f"Unknown visualization type: {vis_type}")
    return save_game_to_html


def random_play_test(num_games: int = 1, vis_type: str = 'table'):
    save_game_to_html = import_visualization(vis_type)
    os.makedirs("random_games", exist_ok=True)

    for game_num in range(num_games):
        game = QuodGame()

        while not game.is_terminal():
            # Place quazars (0 to all remaining)
            num_quazars = random.randint(0, game.quazars[game.current_player])
            for _ in range(num_quazars):
                legal_moves = game.get_legal_moves()
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                game.make_move(move[0], move[1], QUAZAR)

            # Place a quod if possible
            if game.quods[game.current_player] > 0:
                legal_moves = game.get_legal_moves()
                if legal_moves:
                    move = random.choice(legal_moves)
                    quod_type = QUOD_RED if game.current_player == PLAYER_RED else QUOD_BLUE
                    game.make_move(move[0], move[1], quod_type)

        winner = game.get_winner()
        winner_str = "Red" if winner == PLAYER_RED else "Blue" if winner == PLAYER_BLUE else "Draw"

        filename = f"random_games/random_game_{game_num + 1}_{vis_type}.html"
        save_game_to_html(game, filename, "Random Player 1", "Random Player 2")

        print(f"Game {game_num + 1} completed. Winner: {winner_str}")
        print(f"Game log saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play random Quod games")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--vis", type=str, choices=['table', 'svg', 'canvas'], default='table',
                        help="Visualization type")
    args = parser.parse_args()

    random_play_test(args.games, args.vis)
```

