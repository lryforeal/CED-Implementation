# Constrained Exploitability Descent (CED)

This repository contains an offline reinforcement learning algorithm for finding mixed-strategy Nash equilibria in adversarial Markov games. The algorithm is proposed in our paper "Constrained Exploitability Descent: An Offline Reinforcement Learning Method for Finding Mixed-Strategy Nash Equilibrium" [1] at ICML 2025. The game environments and algorithm implementations align with the descriptions in the paper. If you have any questions, please contact lurunyu17@mails.ucas.ac.cn.

The directory "tree_form_game" contains the basic C++ implementation of tabular CED in a small-scale three-stage Markov game. The game starts with Stage 1 and enters Stage 2 or Stage 3 conditioned on previous actions, where Stage 3 is a 5-action Rock-Paper-Scissors-Fire-Water matrix game.

The directory "robotic_combat_game" contains a Python implementation of CED (under a GNN-based representation of multi-agent policy) in a large-scale two-team adversarial game (Team Square vs. Team Circle). The directory also includes the comparative methods of behavior cloning and offline self-play.

<img width="6000" height="3600" alt="combat" src="https://github.com/user-attachments/assets/c242be9f-09b5-4b11-9135-afa025622381" />

Reference:

[1] Runyu Lu, Yuanheng Zhu, and Dongbin Zhao. Constrained exploitability descent: An offline reinforcement learning method for finding mixed-strategy Nash equilibrium. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=unUW6MC7Su.
