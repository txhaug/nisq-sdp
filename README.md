# Noisy intermediate-scale quantum algorithm for semidefinite programming

Program to calculate SDPs using Noisy intermediate scale quantum (NISQ) computers.

Companion code to "Noisy intermediate-scale quantum algorithm for semidefinite programming" by Kishor Bharti, Tobias Haug, Vlatko Vedral and Leong-Chuan Kwek.
Phys. Rev. A 105, 052445
arXiv:2106.03891
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.052445


Solves a SDP using a quantum computer via the NISQ SDP Solver (NSS). The SDP problem is mapped onto a set of ansatz quantum states. The cost function and constraints of the SDP are mapped onto Pauli strings.
Then, the quantum states are measured in respect to the Pauli strings. 
The measurement results lead to another SDP with a reduced size, as large as the ansatz space. This SDP is solved on classical computer.

The Jupyter notebook implements the NSS for calculating the Lovasz Theta number and a non-local Bell game.

NISQSDP.py implements finding Hamiltonian ground state with optional symmetry constraint. Further, can find optimal POVMs for state discrimination using NISQ SDP.
Can be run either using generalized eigenvalue problem (standalone with Python, limited to Hamiltonians) or with SDP (requires installation of Matlab). To solve with SDP requires installation of Matlab, CVXPY and matlab engine for Python. 
Also implements a safe way to solve generalized eigenvalue problem (which is robust against small or negative eigenvalues of the E_matrix) using two regular eigenvalue problems.


By Tobias Haug, Kishor Bharti

Prerequisits:
qutip
cvxpy
