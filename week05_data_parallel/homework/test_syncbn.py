import torch
from syncbn import SyncBatchNorm

import pytest

testdata = [
    (worker, dim, batch) 
    for worker in [1, 4] 
    for dim in [128, 256, 512, 1024] 
    for batch in [32, 64]
]

def _test_batchnorm(input_bn1d, input_sbn, bn1d_queue, sbn_queue):
    bn1d = torch.nn.BatchNorm1d(input_bn1d.size(1))
    bn1d_out = bn1d(input_bn1d)
    
    sbn = SyncBatchNorm(input_sbn.size(1))
    sbn_out = sbn(input_sbn)    
    
    bn1d_queue.put(bn1d_out)
    sbn_queue.put(sbn_out)
    

@pytest.mark.parametrize("num_workers,hid_dim,batch_size", testdata)
def test_batchnorm(num_workers, hid_dim, batch_size):
    # Verify that the implementation of SyncBatchNorm gives the same results (both for outputs
    # and gradients with respect to input) as torch.nn.BatchNorm1d on a variety of inputs.

    # This can help you set up the worker processes. Child processes launched with `spawn` can still run
    # torch.distributed primitives, but you can also communicate their outputs back to the main process to compare them
    # with the outputs of a non-synchronous BatchNorm.
    ctx = torch.multiprocessing.get_context("spawn")
    
    input = torch.randn(batch_size, 32, hid_dim)
    input_copy = input.detach().clone()
    
    input.requires_grad = True
    input_copy.requires_grad = True
    
    chunk_input = torch.chunk(input, chunks=num_workers, dim=1)
    chunk_input_copy = torch.chunk(input_copy, chunks=num_workers, dim=1)
    
    processes = []
    for worker in range(num_workers):
        bn1d_queue = ctx.Queue()
        sbn_queue = ctx.Queue()
        process = ctx.Process(
            target=_test_batchnorm, 
            args=(chunk_input[worker], chunk_input_copy[worker], bn1d_queue, sbn_queue),
        )
        process.start()
        processes.append((process, bn1d_queue, sbn_queue))
        
    bn1d_outputs = []
    sbn_outputs = []
    for process, bn1d_queue, sbn_queue in processes:
        bn1d_outputs.append(bn1d_queue.get())
        sbn_outputs.append(sbn_queue.get())
        process.join()
    
    bn1d_outputs = torch.stack(bn1d_outputs, dim=0)
    sbn_outputs = torch.stack(sbn_outputs, dim=0)
    
    assert torch.allclose(bn1d_outputs, sbn_outputs), "Outputs of SyncBatchNorm are wrong"
    
    bn1d_outputs[:batch_size // 2].sum().backward()
    sbn_outputs[:batch_size // 2].sum().backward()
    
    assert torch.allclose(input.grad, input_copy.grad, atol=1e-3), "grads of SyncBatchNorm are wrong"
    
    