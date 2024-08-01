import html
from quod_game import QUOD_RED, QUOD_BLUE, QUAZAR, EMPTY_CELL, DISABLED_CELL, BOARD_SIZE, PLAYER_RED, PLAYER_BLUE


def board_to_canvas(board, quazar_owners, turn, move, player, last_move=None, winning_squares=None, is_final=False):
    cell_size = 30
    board_size = cell_size * BOARD_SIZE
    canvas_id = f"quodBoard_turn{turn}_move{move}_player{player}"

    canvas_script = f"""
    <canvas id="{canvas_id}" width="{board_size}" height="{board_size}"></canvas>
    <script>
    (function() {{
        const canvas = document.getElementById('{canvas_id}');
        if (!canvas) {{
            console.error('Canvas element not found:', '{canvas_id}');
            return;
        }}
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

        // Draw winning squares
        if ({is_final} && {winning_squares}) {{
            const winning_squares = {winning_squares};
            ctx.strokeStyle = 'yellow';
            ctx.lineWidth = 3;
            for (const square of winning_squares) {{
                ctx.beginPath();
                ctx.moveTo(square[0][1] * cellSize + cellSize/2, square[0][0] * cellSize + cellSize/2);
                for (let i = 1; i < 4; i++) {{
                    ctx.lineTo(square[i][1] * cellSize + cellSize/2, square[i][0] * cellSize + cellSize/2);
                }}
                ctx.closePath();
                ctx.stroke();
            }}
        }}
    }})();
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

                is_final = (turn == len(game.moves) and idx == len(moves) - 1)
                f.write(board_to_canvas(board, quazar_owners, turn, idx, player, last_move=move,
                                        winning_squares=game.winning_square, is_final=is_final))

        winner = game.get_winner()
        winner_str = "Red" if winner == PLAYER_RED else "Blue" if winner == PLAYER_BLUE else "Draw"
        f.write(
            f"<h2>Final Result: <span style='color: {'red' if winner == PLAYER_RED else 'blue' if winner == PLAYER_BLUE else 'black'};'>{winner_str}</span></h2>")

        f.write("</body></html>")