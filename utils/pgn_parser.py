# utils/pgn_parser.py

import chess.pgn

def extract_moves_from_pgn(pgn_path, player_name=None, max_games=None):
    """
    Extract (FEN, move, color) from a PGN file.
    
    Args:
        pgn_path (str): Path to the PGN file.
        player_name (str): Your username, e.g. "beginner1937". If None, all moves are included.
        max_games (int): Max number of games to parse.
    
    Returns:
        List of tuples: (FEN before move, move in UCI format, color)
    """
    data = []
    with open(pgn_path, "r", encoding="utf-8") as f:
        game_count = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if max_games and game_count >= max_games:
                break
            
            board = game.board()
            white = game.headers["White"]
            black = game.headers["Black"]

            # Decide which color is the player
            if player_name is None:
                player_color = None
            elif white == player_name:
                player_color = chess.WHITE
            elif black == player_name:
                player_color = chess.BLACK
            else:
                continue  # skip game if player didn't play

            for node in game.mainline():
                move = node.move
                if move is None:
                    continue
                fen = board.fen()
                color = board.turn  # who's about to move
                uci = move.uci()

                # If filtering by player
                if player_color is None or player_color == color:
                    data.append((fen, uci, "white" if color == chess.WHITE else "black"))

                board.push(move)

            game_count += 1

    return data
