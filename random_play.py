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