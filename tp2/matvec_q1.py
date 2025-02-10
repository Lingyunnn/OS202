from mpi4py import MPI
import numpy as np
from time import time

# Dimension du problème (peut-être changé)
dim = 5760 #(1,2,3,4,5,6)
# Initialisation de la matrice
A = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(dim)])

# Initialisation du vecteur u
u = np.array([i + 1. for i in range(dim)])

# Récupération de l'environnement MPI, de l'ID du processus et du nombre total de processus
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Identifiant du processus courant
size = comm.Get_size()  # Nombre total de processus

# Calcul du nombre de colonnes à traiter par chaque processus
Nloc = dim // size

# Définition des colonnes à traiter pour chaque processus
start_col = rank * Nloc
end_col = (rank + 1) * Nloc

# Calcul du produit matrice-vecteur partiel pour chaque processus
comm.Barrier()
deb = time()
partial_result = np.zeros(dim)
for j in range(start_col, end_col):
    partial_result += A[:, j] * u[j]

# Rassemblement des résultats partiels vers le processus racine (rank 0)
if rank == 0:
    result = np.copy(partial_result)
    for i in range(1, size):
        # Réception des résultats partiels des autres processus
        partial = comm.recv(source=i, tag=99)
        result += partial
    print(f"v = {result}")
    fin = time()
    total_time = fin - deb
    print(f"Total execution time with {size} processes: {total_time:.4f} seconds")
else:
    # Envoi du résultat partiel au processus racine
    comm.send(partial_result, dest=0, tag=99)
