# CS440-Project3

This project implements a Markov Decision Process (MDP) to optimize a bot’s navigation strategy for capturing a randomly moving rat within a dynamically generated D × D grid-based spaceship environment. The bot has full knowledge of the grid layout and rat position, while the rat moves stochastically.

The solution involves two key components:

1. MDP-based Value Iteration – Computes the optimal policy that minimizes the expected number of moves to capture the rat.

2. Machine Learning Models – Trains neural networks and Convolutional Neural Networks (CNNs) to approximate the MDP value function and predict expected capture steps.

The project explores the trade-offs between MDP computation and model-based approximation for real-time decision-making, demonstrating how spatial feature representation via CNNs significantly improves predictive accuracy and generalization across multiple grid layouts.

Please refer to Report.pdf for a detailed analysis of the project and results.
