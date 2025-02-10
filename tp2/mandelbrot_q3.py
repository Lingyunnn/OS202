from mpi4py import MPI
import numpy as np
from PIL import Image
from time import time
from dataclasses import dataclass
import matplotlib.cm

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        z: complex
        if c.real*c.real + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1) + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.E-14)):
                return self.max_iterations
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - np.log(np.log(abs(z))) / np.log(2)
                return iter
        return self.max_iterations

# MPI initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#  Image parameters
width, height = 1024, 1024
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
scaleX = 3.0 / width
scaleY = 2.25 / height

if rank == 0:
    start_time = time()
    
    # Main process: assigning tasks
    task_queue = list(range(width))  #Task queue storing all row indexes
    num_slaves = size - 1
    active_workers = min(num_slaves, len(task_queue))
    
    # Storing the final result
    final_result = np.zeros((width, height), dtype=np.double)

    # Sending initial task
    for worker in range(1, active_workers + 1):
        row = task_queue.pop(0)
        comm.send(row, dest=worker, tag=1)

    completed_tasks = 0
    while completed_tasks < width:
        # Receiving Data
        data = comm.recv(source=MPI.ANY_SOURCE, tag=2)
        source_rank = data["rank"]
        row_idx = data["row"]
        final_result[row_idx, :] = data["result"]
        completed_tasks += 1

        # Assign new tasks
        if task_queue:
            new_row = task_queue.pop(0)
            comm.send(new_row, dest=source_rank, tag=1)
        else:
            comm.send(None, dest=source_rank, tag=0)
    
    # Execution time
    end_time = time()
    total_time = end_time - start_time

    print(f"Total execution time with {size - 1} processes: {total_time:.4f} seconds")

    # Process the image
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_result.T) * 255))
    image.save("mandelbrot_optimized.png")

else:
    while True:
       # Receive tasks from the process 0
        row_idx = comm.recv(source=0, tag=MPI.ANY_TAG)
        if row_idx is None:
            break 
        
        # Compute the Mandelbrot row
        local_result = np.zeros(height, dtype=np.double)
        for y in range(height):
            c = complex(-2.0 + scaleX * row_idx, -1.125 + scaleY * y)
            local_result[y] = mandelbrot_set.convergence(c, smooth=True)

        # Send results back to the process 0
        comm.send({"rank": rank, "row": row_idx, "result": local_result}, dest=0, tag=2)
