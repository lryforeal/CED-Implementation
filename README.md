# Constrained Exploitability Descent (CED)

This repository contains an offline reinforcement learning algorithm for finding mixed-strategy Nash equilibria in adversarial Markov games. The algorithm is proposed in the paper "Constrained Exploitability Descent: An Offline Reinforcement Learning Method for Finding Mixed-Strategy Nash Equilibrium" [1] at ICML 2025. The game environments and algorithm implementations align with the descriptions in the paper. If you have any questions, please contact lurunyu17@mails.ucas.ac.cn.

The directory "tree_form_game" contains the basic C++ implementation of tabular CED in this 3-stage small-scale Markov game.

The directory "robotic_combat_game" contains our Python implementation of CED with graph neural networks in this large-scale two-team adversarial game. This directory also contains the comparative methods of behavior cloning and offline self-play [2].

Reference:

[1] Runyu Lu, Yuanheng Zhu, and Dongbin Zhao. Constrained exploitability descent: An offline reinforcement learning method for finding mixed-strategy Nash equilibrium. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=unUW6MC7Su.

[2] Jingxiao Chen, Weiji Xie, Weinan Zhang, Ying Wen, et al. Offline fictitious self-play for competitive games. arXiv preprint arXiv:2403.00841, 2024.
