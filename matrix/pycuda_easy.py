import math

import numpy as np
import pycuda.autoinit
from pycuda import driver as drv
from pycuda.compiler import SourceModule

from matrix.common import measure_time, increment_by_one_cpu, add_compiler_to_PATH


add_compiler_to_PATH()

mod = SourceModule("""
__global__ void increment_by_one(float *array){
  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  array[index]++;
}
""")

increment_by_one = mod.get_function('increment_by_one')

an_array = np.zeros((100_000, )).astype(np.float32)
an_array_copy = an_array.copy()

print('[' + '.' * 30 + 'GPU' + '.' * 30 + ']')
treads_pre_block = 64
block_pre_grid = math.ceil(an_array.size / treads_pre_block)

for _ in range(3):
    with measure_time('GPU'):
        increment_by_one(
            drv.InOut(an_array),
            block=(treads_pre_block, 1, 1),
            grid=(block_pre_grid, 1),
        )

print('[' + '.' * 30 + 'CPU' + '.' * 30 + ']')
for _ in range(3):
    with measure_time('CPU'):
        increment_by_one_cpu(an_array_copy)

print(f'GPU: {an_array}')
print(f'CPU: {an_array_copy}')
