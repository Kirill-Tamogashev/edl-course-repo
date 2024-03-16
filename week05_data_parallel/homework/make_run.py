import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from syncbn import SyncBatchNorm

def run(rank, size, input):
    """ Distributed function to be implemented later. """

    micro_net = nn.Sequential([
        nn.Linear(in_features=100, out_features=100),
        # SyncBatchNorm(100),
        # nn.Linear(in_features=100, out_features=100)
    ])
    # sbn = SyncBatchNorm(100)
    # out = sbn(input)

    out = micro_net(input)
    print(f"Output shape: {out.shape}, {rank=}")

    out_sum = out.sum()
    out_sum.backward()


def init_process(rank, size, input, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, input)


if __name__ == '__main__':
    size = 2
    x = torch.rand(size, 16, 100, 256, requires_grad=True)
    torch.set_num_threads(1)

    processes = []
    mp.set_start_method("spawn")
    print("before")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, x[rank], run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
