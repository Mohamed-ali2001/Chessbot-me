# bot/bot_player.py

import numpy as np
import chess
import chess.engine
import tensorflow as tf
from utils.featurizer import fen_to_tensor
from utils.encoder import index_to_move, move_to_index, UCI_MOVES

# Load model
model = tf.keras.models.load_model("model/chess_model.keras")

def predict_move(board: chess.Board, top_k=5):
    """
    Predict the most likely legal move using the trained model.
    """
    fen = board.fen()
    x = fen_to_tensor(fen)
    x = np.expand_dims(x, axis=0)  # shape: (1, 8, 8, 12)

    probs = model.predict(x, verbose=0)[0]  # shape: (num_moves,)
    legal_moves = list(board.legal_moves)
    print(legal_moves)
    # Filter only legal UCI moves with their probabilities
    legal_uci = [move.uci() for move in legal_moves]
    legal_probs = [(m, probs[move_to_index(m)]) for m in legal_uci if move_to_index(m) != -1]
    legal_probs.sort(key=lambda x: x[1], reverse=True)

    if not legal_probs:
        return None  # No legal moves available

    return chess.Move.from_uci(legal_probs[0][0])  # return the top legal move

def play_human_vs_bot(bot_color="black"):
    board = chess.Board()
    print("Starting a new game! Type moves in UCI (e.g., e2e4)")

    print(board, "\n")

    while not board.is_game_over():
        if (board.turn == chess.WHITE and bot_color == "white") or \
           (board.turn == chess.BLACK and bot_color == "black"):
            # Bot's turn
            move = predict_move(board)
            if move is None:
                print("Bot resigns (no legal moves)")
                break
            print(f"Bot plays: {move}")
        else:
            # Human's turn
            move = None
            while move not in board.legal_moves:
                user_input = input("Your move: ")
                try:
                    move = chess.Move.from_uci(user_input)
                    if move not in board.legal_moves:
                        print("Illegal move. Try again.")
                except:
                    print("Invalid format. Use UCI like e2e4.")
        
        board.push(move)
        print("\n", board, "\n")

    print("Game over.")
    print("Result:", board.result())


if __name__ == "__main__":
    # You play as White → bot plays Black
    play_human_vs_bot(bot_color="black")
