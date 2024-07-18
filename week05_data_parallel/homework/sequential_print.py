import os

import torch.distributed as dist
from torch.multiprocessing import Process


def run_sequential(rank, size, num_iter=10):
    """
    Prints the process rank sequentially according to its number over `num_iter` iterations,
    separating the output for each iteration by `---`
    Example (3 processes, num_iter=2):
    ```
    Process 0
    Process 1
    Process 2
    ---
    Process 0
    Process 1
    Process 2
    ```
    """

    for _ in range(num_iter):
        for curr_rank in range(size):
            if rank == curr_rank:
                print("Process ", rank, flush=True)
            dist.barrier()
        
        if rank == 0:
            print(" --- ", flush=True)
        dist.barrier()


def init_process(rank, size, fn, master_port, backend='gloo'):
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)
    

        
if __name__ == "__main__":
    size = 5
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29005)
    os.environ['WORLD_SIZE'] = str(size)
    
    # run_sequential(local_rank, dist.get_world_size())
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_sequential, 29005))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
