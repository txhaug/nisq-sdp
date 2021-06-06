# NISQ Algorithm for Semidefinite Programming

Program to calculate SDPs using Noisy intermediate scale quantum (NISQ) computers.

Decomposes a classical SDP problem into Pauli strings. 
Then, a set of ansatz quantum states are used to measure overlaps, which are then used to solve the original SDP via another SDP.

By Tobias Haug, Kishor Bharti

Prerequisits:
qutip
cvxpy
