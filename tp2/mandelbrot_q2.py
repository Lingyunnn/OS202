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
        iter: int
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

# Image parameters
width, height = 1024, 1024
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
scaleX = 3.0 / width
scaleY = 2.25 / height

# Process 0 precomputes load balancing
if rank == 0:
    estimated_load = np.zeros(width, dtype=np.float64)
    for x in range(width):
        for y in range(height):
            c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
            if c.real*c.real + c.imag*c.imag < 0.0625 or (c.real+1)*(c.real+1) + c.imag*c.imag < 0.0625 or ((c.real > -0.75) and (c.real < 0.5) and abs(c - 0.25) < 0.5 * (1 - (c.real - 0.25) / max(abs(c - 0.25), 1.E-14))):
                estimated_load[x] += 1  # Low complexity
            else:
                estimated_load[x] += 50  # High complexity
    cumulative_load = np.cumsum(estimated_load)
    total_work = cumulative_load[-1]
    row_split = np.searchsorted(cumulative_load, np.linspace(0, total_work, size + 1))
else:
    row_split = None

row_split = comm.bcast(row_split, root=0)
start_row, end_row = row_split[rank], row_split[rank + 1]

# Compute Mandelbrot rows locally
local_result = np.zeros((end_row - start_row, height), dtype=np.double)
deb = time()
for x in range(start_row, end_row):
    for y in range(height):  # 注意这里应该是 height 而不是 width
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
        local_result[x - start_row, y] = mandelbrot_set.convergence(c, smooth=True)

# Gather results
if rank == 0:
    final_result = np.zeros((width, height), dtype=np.double)
    recvcounts = [(row_split[i+1] - row_split[i]) * height for i in range(size)]
    displacements = [sum(recvcounts[:i]) for i in range(size)]
else:
    final_result = None
    recvcounts = None
    displacements = None

comm.Gatherv(local_result, (final_result, recvcounts, displacements, MPI.DOUBLE), root=0)

# Compute speedup
if rank == 0:
    fin = time()
    total_time = fin - deb
    print(f"Total execution time with {size} processes: {total_time:.4f} seconds")
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_result.T)*255))
    image.save("mandelbrot_optimized.png")
