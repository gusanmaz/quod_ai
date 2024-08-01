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


