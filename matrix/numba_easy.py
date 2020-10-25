import numpy as np
from numba import cuda

from matrix.common import measure_time, increment_by_one_cpu


@cuda.jit
def increment_by_one(array: np.ndarray):
    """
    Increment all array by one
    :param array: array
    """

    # Tread id in an 1D block
    tx: int = cuda.threadIdx.x

    # Block id in an 1D grid
    ty: int = cuda.blockIdx.x

    # Block width, i.e. number of threads per block
    bw: int = cuda.blockDim.x

    pos = tx + ty * bw
    if pos < an_array.size:
        array[pos] += 1


# create an array
an_array = np.array(range(100_000))
an_array_copy = an_array.copy()
print(an_array)

# blocks and treads
treads_pre_block = 64
blocks_pre_grid = (an_array.size + (treads_pre_block - 1)) // treads_pre_block


print('[' + '.' * 30 + 'GPU' + '.' * 30 + ']')
for _ in range(3):
    with measure_time('GPU'):
        # run kernel
        increment_by_one[blocks_pre_grid, treads_pre_block](an_array)

print('[' + '.' * 30 + 'CPU' + '.' * 30 + ']')
for _ in range(3):
    with measure_time('CPU'):
        increment_by_one_cpu(an_array_copy)


print(f'GPU: {an_array}')
print(f'CPU: {an_array_copy}')
