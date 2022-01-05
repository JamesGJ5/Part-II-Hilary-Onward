from mpi4py import MPI
print("Hello World (from process %d)" % MPI.COMM_WORLD.Get_rank())