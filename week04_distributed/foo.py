import time 
import numpy as np


def foo(i):
    """ Imagine particularly computation-heavy function... """
    print(end=f"Began foo({i})...\n")
    result = np.sin(i)
    time.sleep(abs(result))
    print(end=f"Finished foo({i}) = {result:.3f}.\n")
    return result


def compute_and_send(i, output_pipe):
    print(end=f"Began compute_and_send({i})...\n")
    result = np.sin(i)
    time.sleep(abs(result))
    print(end=f"Finished compute_and_send({i}) = {result:.3f}.\n")

    output_pipe.send(result)
