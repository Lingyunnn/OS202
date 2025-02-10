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

# Distribute rows across processes
rows_per_proc = width // size
start_row = rank * rows_per_proc
end_row = (rank + 1) * rows_per_proc if rank != size - 1 else width
local_result = np.zeros((end_row - start_row, height), dtype=np.double)

recvcounts = [rows_per_proc * height] * size
displacements = [i * rows_per_proc * height for i in range(size)]
recvcounts[-1] = (width - (size - 1) * rows_per_proc) * height 

deb = time()
for x in range(start_row, end_row):
    for y in range(height):
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y)
        local_result[x - start_row, y] = mandelbrot_set.convergence(c, smooth=True)

# Gather results on process 0
if rank == 0:
    final_result = np.zeros((width, height), dtype=np.double)
    fin = time()
    total_time = fin - deb
    print(f"Total execution time with {size} processes: {total_time:.4f} seconds")
else:
    final_result = None

comm.Gatherv(local_result, (final_result, recvcounts, displacements, MPI.DOUBLE), root=0)

# Process 0 process the image
if rank == 0:
    deb = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_result.T)*255))
    fin = time()
    print(f"Image creation time : {fin-deb}")
    image.save("mandelbrot_optimized.png")
