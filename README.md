# Neural Quantum States for the Schwinger Model

Research project (2020–2021) exploring neural network ansätze for variational ground state search in a 1D lattice Schwinger model (quantum electrodynamics on a lattice).

Presented at **ICQT 2021** (International Conference on Quantum Technologies) — see `ICQT2021_39.pptx`.

## What this is

The Schwinger model Hamiltonian (mass term + XY hopping + interaction) is constructed exactly for small system sizes (N = 8–18 qubits). Several neural network architectures are trained as variational wavefunctions ψ(σ) to minimize the ground state energy ⟨ψ|H|ψ⟩ via gradient descent (Variational Monte Carlo).

## Models compared

| Notebook | Architecture | Best result |
|---|---|---|
| `Autoregressive_tanh_l.ipynb` | Autoregressive MLP (masked linear) | Matches exact (N=8) |
| `RNN.ipynb` | CNN + LSTM | Matches exact (N=8) |
| `MADE.ipynb` | MADE (masked autoencoder) | ~3% error (N=8) |
| `feed_forward.ipynb` | Feed-forward MLP | ~14% error (N=14) |
| `Big_feed_forward.ipynb` | Feed-forward MLP | (N=16) |
| `CNN.ipynb` | 1D Convolutional network | (N=14) |
| `Autoencoder.ipynb` | Convolutional autoencoder | Matches exact (N=10) |
| `sparce_diagonalization.ipynb` | Sparse exact diagonalization | Reference energies |
| `Check_H.ipynb` | Exact ground state energies | N = 8–20 |

## Key files

- `Hamiltonian.py` — builds the full Hamiltonian matrix via tensor products of Pauli matrices
- `build_H.py` — batched, GPU-friendly Hamiltonian matrix element computation
- `number_of_qubits_12.npy` — precomputed eigenvector for N=12 (used as initialization)
- `Draw_all.ipynb` — plots and comparison of all models

## Setup

All notebooks were developed in Google Colab (2020–2021). Main dependencies: `torch`, `numpy`, `sympy`, `scipy`, `matplotlib`.
