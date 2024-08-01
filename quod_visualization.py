import html
from quod_game import QUOD_RED, QUOD_BLUE, QUAZAR, EMPTY_CELL, DISABLED_CELL, BOARD_SIZE, PLAYER_RED, PLAYER_BLUE

def board_to_html(board, last_move=None, winning_square=None, is_final_state=False):
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

            if is_final_state and winning_square and (i, j) in winning_square:
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

                f.write(board_to_html(board, move, game.winning_square, is_final_state=False))

        winner = game.get_winner()
        winner_str = "Red" if winner == PLAYER_RED else "Blue" if winner == PLAYER_BLUE else "Draw"
        f.write(
            f"<h2>Final Result: <span style='color: {'red' if winner == PLAYER_RED else 'blue' if winner == PLAYER_BLUE else 'black'};'>{winner_str}</span></h2>")

        # Display final board state with winning square highlighted
        f.write(board_to_html(board, None, game.winning_square, is_final_state=True))

        f.write("</body></html>")