# Snake Game with Reinforcement Learning

This project implements a deep Q-learning (DQN) agent to play the classic Snake game using reinforcement learning. The snake learns to navigate the grid, collect food, and maximize its survival time.

## Features

- **Deep Q-Network (DQN)**: Uses a neural network to approximate Q-values for optimal decision-making.
- **Experience Replay**: Stores past experiences in memory to improve learning stability.
- **Target Network**: Uses a separate target network to enhance learning.
- **8-Direction Vision**: The agent perceives obstacles, food, and itself in 8 directions for better awareness.
- **Adaptive Rewards**: Encourages eating food while penalizing prolonged survival without progress.
- **Wall Wrapping Mode**: Snake wraps around screen edges instead of dying, allowing diverse training.
- **Starvation Mechanism**: The agent is penalized if it fails to eat food within a given step limit.

## Installation

1. Install Python dependencies:
   ```sh
   pip install pygame numpy torch
   ```
2. Run the training script:
   ```sh
   python snake.py
   ```

## Training Process

- The agent starts with random movements and learns over time through reinforcement learning.
- Each episode consists of a game session where the agent plays until it dies or reaches the step limit.
- The agent updates its strategy using past experiences and reward feedback.
- Training can take several episodes to improve performance.

## Future Improvements

- Implementing recurrent neural networks (RNNs) for better memory.
- Enhancing reward functions for more efficient learning.
- Optimizing neural network architecture for better performance.
- Experimenting with different reinforcement learning algorithms (e.g., PPO, A2C).

## Author

Developed by [mezian fellag]

