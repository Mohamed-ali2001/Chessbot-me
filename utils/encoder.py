# utils/encoder.py

import chess
import itertools

# Generate all possible UCI moves for a standard chess board
def generate_uci_move_list():
    board = chess.Board()
    all_moves = set()

    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            move = chess.Move(from_sq, to_sq)
            all_moves.add(move.uci())

            # Add promotion moves
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_sq, to_sq, promotion=promo)
                if chess.square_rank(to_sq) in [0, 7]:
                    all_moves.add(promo_move.uci())

    return sorted(all_moves)

# UCI move vocabulary and lookup tables
UCI_MOVES = generate_uci_move_list()
MOVE_TO_INDEX = {move: i for i, move in enumerate(UCI_MOVES)}
INDEX_TO_MOVE = {i: move for move, i in MOVE_TO_INDEX.items()}

# Convert move in UCI format to index
def move_to_index(move_uci: str) -> int:
    return MOVE_TO_INDEX.get(move_uci, -1)  # -1 if not found

# Convert index to UCI move
def index_to_move(index: int) -> str:
    return INDEX_TO_MOVE.get(index, "0000")
