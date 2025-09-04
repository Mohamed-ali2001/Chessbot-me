# data/build_dataset.py

import os
import numpy as np
import tensorflow as tf
from utils.pgn_parser import extract_moves_from_pgn
from utils.featurizer import fen_to_tensor
from utils.encoder import move_to_index
from tqdm import tqdm



BATCH_SIZE = 10000

def build_dataset_multi_batch(pgn_paths, player_names, output_path, max_games=None):
    print("Parsing PGNs and extracting moves in batches...")
    all_samples = []
    for pgn_path, player_name in tqdm(zip(pgn_paths, player_names), total=len(pgn_paths), desc="Parsing PGN files"):
        samples = extract_moves_from_pgn(pgn_path, player_name=player_name, max_games=max_games)
        all_samples.extend(samples)

    X_batch, y_batch = [], []
    batch_idx = 0
    total_samples = 0
    for i, (fen, move_uci, color) in enumerate(tqdm(all_samples)):
        board_tensor = fen_to_tensor(fen)
        move_idx = move_to_index(move_uci)
        if move_idx == -1:
            continue
        X_batch.append(board_tensor)
        y_batch.append(move_idx)
        if len(X_batch) == BATCH_SIZE:
            np.savez_compressed(f"data/train_data_batch_{batch_idx}.npz", X=np.array(X_batch, dtype=np.float32), y=np.array(y_batch, dtype=np.int64))
            print(f"Saved batch {batch_idx} with {len(X_batch)} samples.")
            total_samples += len(X_batch)
            X_batch, y_batch = [], []
            batch_idx += 1
    # Save any remaining samples
    if X_batch:
        np.savez_compressed(f"data/train_data_batch_{batch_idx}.npz", X=np.array(X_batch, dtype=np.float32), y=np.array(y_batch, dtype=np.int64))
        print(f"Saved batch {batch_idx} with {len(X_batch)} samples.")
        total_samples += len(X_batch)

    # Merge all batches into one file
    X_all, y_all = [], []
    for i in range(batch_idx + 1):
        batch = np.load(f"data/train_data_batch_{i}.npz")
        X_all.append(batch["X"])
        y_all.append(batch["y"])
        os.remove(f"data/train_data_batch_{i}.npz")
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    print(f"Saving {total_samples} samples to {output_path}...")
    np.savez_compressed(output_path, X=X_all, y=y_all)

if __name__ == "__main__":
    build_dataset_multi_batch(
        pgn_paths=["data/chess_com_games_2024-10-11.pgn", "data/lichess_db_standard_rated_2014-10.pgn"],
        player_names=["beginner1937", None],
        output_path="data/train_data.npz",
        max_games=None
    )
