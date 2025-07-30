# utils/featurizer.py

import numpy as np
import chess

# 12 channels: white/black × [P, N, B, R, Q, K]
PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def fen_to_tensor(fen: str) -> np.ndarray:
    """
    Converts a FEN string into an 8x8x12 tensor.

    Args:
        fen (str): FEN string representing the board position.

    Returns:
        np.ndarray: A binary tensor of shape (8, 8, 12)
    """
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, 12), dtype=np.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = 7 - chess.square_rank(square)  # flip vertically for top-down board
            col = chess.square_file(square)
            idx = PIECE_TO_INDEX[piece.symbol()]
            tensor[row, col, idx] = 1.0

    return tensor
