# data/build_dataset.py

import numpy as np
from utils.pgn_parser import extract_moves_from_pgn
from utils.featurizer import fen_to_tensor
from utils.encoder import move_to_index
from tqdm import tqdm 

def build_dataset(pgn_path, player_name, output_path, max_games=None):
    print("Parsing PGN and extracting moves...")
    samples = extract_moves_from_pgn(pgn_path, player_name=player_name, max_games=max_games)

    X = []
    y = []

    for fen, move_uci, color in tqdm(samples):
        board_tensor = fen_to_tensor(fen)
        move_idx = move_to_index(move_uci)

        # Only include known moves
        if move_idx == -1:
            continue

        X.append(board_tensor)
        y.append(move_idx)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"Saving {len(X)} samples to {output_path}...")
    np.savez_compressed(output_path, X=X, y=y)

if __name__ == "__main__":
    build_dataset(
        pgn_path="data/chess_com_games_2024-10-11.pgn",
        player_name="beginner1937",
        output_path="data/train_data.npz",
        max_games=None  # Set to 10 if you want to limit for testing
    )
