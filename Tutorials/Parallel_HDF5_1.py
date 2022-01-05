from mpi4py import MPI
import time

time.sleep(5)

# First I shall test MPI from mpi4py
print("Hello World (from process %d)" % MPI.COMM_WORLD.Get_rank())

# print("Hello")