from mpi4py import MPI
import numpy as np
from time import time

# Dimension du problème
dim = 5760  #(1,2,3,4,5,6)

# Initialisation de la matrice et du vecteur
A = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(dim)])
u = np.array([i + 1. for i in range(dim)])

# Initialisation de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Calcul du nombre de lignes par processus
Nloc = dim // size
start_row = rank * Nloc
end_row = (rank + 1) * Nloc

# Chaque processus travaille sur un sous-ensemble de lignes
deb = time()
local_result = np.zeros(Nloc)

for i in range(start_row, end_row):
    local_result[i - start_row] = np.dot(A[i, :], u)

# Rassemblement des résultats
if rank == 0:
    result = np.zeros(dim)
    result[start_row:end_row] = local_result
    for i in range(1, size):
        recv_data = np.zeros(Nloc)
        comm.Recv(recv_data, source=i, tag=99)
        start = i * Nloc
        end = (i + 1) * Nloc
        result[start:end] = recv_data
    print(f"v = {result}")
    fin = time()
    print(f"Total execution time with {size} processes: {fin - deb:.4f} seconds")
else:
    comm.Send(local_result, dest=0, tag=99)