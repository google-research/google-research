import numpy as np
from mpi4py import MPI

ROOT_PROC_RANK = 0

def get_num_procs():
    return MPI.COMM_WORLD.Get_size()

def get_proc_rank():
    return MPI.COMM_WORLD.Get_rank()

def is_root_proc():
    rank = get_proc_rank()
    return rank == ROOT_PROC_RANK

def bcast(x):
    MPI.COMM_WORLD.Bcast(x, root=ROOT_PROC_RANK)
    return

def reduce_sum(x):
    return reduce_all(x, MPI.SUM)

def reduce_prod(x):
    return reduce_all(x, MPI.PROD)

def reduce_avg(x):
    buffer = reduce_sum(x)
    buffer /= get_num_procs()
    return buffer

def reduce_min(x):
    return reduce_all(x, MPI.MIN)

def reduce_max(x):
    return reduce_all(x, MPI.MAX)

def reduce_all(x, op):
    is_array = isinstance(x, np.ndarray)
    x_buf = x if is_array else np.array([x])
    buffer = np.zeros_like(x_buf)
    MPI.COMM_WORLD.Allreduce(x_buf, buffer, op=op)
    buffer = buffer if is_array else buffer[0]
    return buffer

def gather_all(x):
    is_array = isinstance(x, np.ndarray)
    x_buf = np.array([x])
    buffer = np.zeros_like(x_buf)
    buffer = np.repeat(buffer, get_num_procs(), axis=0)
    MPI.COMM_WORLD.Allgather(x_buf, buffer)
    buffer = list(buffer)
    return buffer