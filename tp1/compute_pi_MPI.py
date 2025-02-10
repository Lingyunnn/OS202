from mpi4py import MPI
import numpy as np
import time

# Initialisation MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Nombre total d'échantillons :
nb_samples = 100000000
samples_per_proc = nb_samples // size

def monte_carlo_pi(samples):
    x = 2. * np.random.random_sample((samples,)) - 1.
    y = 2. * np.random.random_sample((samples,)) - 1.
    return np.sum(x * x + y * y < 1)

beg = time.time()
local_count = monte_carlo_pi(samples_per_proc)

total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

if rank == 0:
    approx_pi = 4. * total_count / nb_samples
    end = time.time()
    
    print(f"Temps pour calculer pi : {end - beg} secondes")
    print(f"Pi vaut environ {approx_pi}")