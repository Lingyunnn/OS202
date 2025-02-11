import numpy as np
import time
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()
name = MPI.Get_processor_name()

N = 100

if len(sys.argv) > 1:
    N = int(sys.argv[1])

filename = f"Output{rank:03d}.txt"
out = open(filename, mode='w')

reste = N % nbp
NLoc = N // nbp + (1 if reste > rank else 0)
out.write(f"Nombre de valeurs locales : {NLoc}\n")

values = np.random.randint(-32768, 32768, size=NLoc, dtype=np.int64)
out.write(f"Valeurs initiales of bucket {rank}: {values}\n")

debut = time.time()
values = np.sort(values)

# Calculate nbp+1 quantiles
quantiles = np.quantile(values, np.linspace(0, 1, nbp + 1))

# Send quantiles to process 0
if rank != 0:
    comm.Send([quantiles, MPI.DOUBLE], dest=0)

if rank == 0:
    all_quantiles = [quantiles]
    for i in range(1, nbp):
        recv_data = np.empty(nbp + 1, dtype=np.float64)
        comm.Recv([recv_data, MPI.DOUBLE], source=i)
        all_quantiles.append(recv_data)

    all_quantiles = np.concatenate(all_quantiles)
    all_quantiles = np.sort(all_quantiles)
    all_quantiles = np.quantile(all_quantiles, np.linspace(0, 1, nbp + 1))

    # Broadcast quantiles to other processes
    for i in range(1, nbp):
        comm.Send([all_quantiles, MPI.DOUBLE], dest=i)

if rank != 0:
    all_quantiles = np.empty(nbp + 1, dtype=np.float64)
    comm.Recv([all_quantiles, MPI.DOUBLE], source=0)

# Divide the data into buckets
buckets = [[] for _ in range(nbp)]
for num in values:
    for i in range(nbp):
        if all_quantiles[i] <= num < all_quantiles[i + 1]:
            buckets[i].append(num)

# Send bucketed data
recv_data = np.array(buckets[rank], dtype=np.int64)

for i in range(nbp):
    if i != rank:
        bucket_data = np.array(buckets[i], dtype=np.int64)
        bucket_size = np.array([len(bucket_data)], dtype=np.int64)
        
        # Send size first, then data
        comm.Send(bucket_size, dest=i)
        if bucket_size[0] > 0:
            comm.Send(bucket_data, dest=i)

        size_buffer = np.empty(1, dtype=np.int64)
        comm.Recv(size_buffer, source=i)
        recv_size = size_buffer[0]

        if recv_size > 0:
            received_bucket = np.empty(recv_size, dtype=np.int64)
            comm.Recv(received_bucket, source=i)
            recv_data = np.concatenate((recv_data, received_bucket))

recv_data.sort()

# Process 0 collects sorted data
if rank != 0:
    size_to_send = np.array([len(recv_data)], dtype=np.int64)
    comm.Send(size_to_send, dest=0)
    if size_to_send[0] > 0:
        comm.Send(recv_data, dest=0)
else:
    sorted_data = recv_data
    for i in range(1, nbp):
        recv_size = np.empty(1, dtype=np.int64)
        comm.Recv(recv_size, source=i)
        if recv_size[0] > 0:
            received_data = np.empty(recv_size[0], dtype=np.int64)
            comm.Recv(received_data, source=i)
            sorted_data = np.concatenate((sorted_data, received_data))

fin = time.time()

out.write(f"Temps pour le tri : {fin - debut} secondes\n")
if rank == 0:
    out.write(f"Sorted data : {sorted_data}\n")

out.close()
