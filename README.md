# EN4554 Assignment 2 — Convolutional Neural Networks

Manujaya U.G.P. (220386H)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manujayaugp/EN4554-A02/blob/main/220386H_a02.ipynb)

A PyTorch notebook on CIFAR-10 (45k train / 5k validation split):

1. **Backpropagation:** Linear, ReLU and softmax cross-entropy with hand-written forward/backward passes, gradient checks against autograd and finite differences, an MLP trained with SGD, and per-layer gradient norms with sigmoid vs ReLU.
2. **Optimisers:** SGD with momentum and Adam implemented as `torch.optim.Optimizer` subclasses, trajectories on an ill-conditioned quadratic, and a learning-rate study on a small CNN.
3. **Regularisation and residual connections:** weight decay, dropout, flip + crop, cutout and mixup on 5k images; plain vs residual networks of depth 8, 20 and 56.
4. **Best CNN:** a compact ResNet (≤ 500k parameters, ≤ 20 epochs) with ablations and a single test-set evaluation.

Run on Google Colab with a T4 GPU: **Runtime → Run all**.
