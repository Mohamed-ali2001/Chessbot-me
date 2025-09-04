# Chessbot-me

A neural network-based chess bot that predicts moves from board positions and plays against humans using real chess data.

## Overview
This project uses deep learning to train a chess move prediction model from PGN game data (my own games and other games found on kaggle). The bot can play chess against a human, making moves based on the trained model.
NB: the initial goal of this project was to train the bot to play exactly like me but i didn't have enough games to get a better accuracy that's why i added games from kaggle 

## How It Works
- **Data Preparation:** PGN files are parsed to extract board positions and moves. These are converted into tensors and move indices for training.
- **Model:** A convolutional neural network (CNN) is trained to predict the most likely move given a board position.
- **Bot Logic:** The bot loads the trained model and, for each position, predicts the best legal move. The game loop alternates between human and bot moves.

## Setup & Usage
1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare data:** Place your PGN files in the `data/` folder. Run `build_dataset.py` to create the training dataset.
3. **Train the model:** Run `train_model.py` to train and save the model.
4. **Play against the bot:** Run `bot_player.py` and follow the prompts.

## Limitations
- The model only predicts moves seen in the training data; rare or creative moves may not be chosen.
- Training and data processing can be slow and memory-intensive for large PGN files.
- The bot's strength depends on the quality and diversity of the training data.
- No search or evaluation logic—purely move prediction, not full chess engine strength.

## Credits
- Uses [python-chess](https://python-chess.readthedocs.io/) for board and PGN handling.
- Model built with TensorFlow/Keras.
- Data sourced from chess.com and Kaggle.

## References
- [Stockfish](https://stockfishchess.org/)
- [Kaggle Chess Datasets](https://www.kaggle.com/datasets)

---
Feel free to contribute or suggest improvements!
