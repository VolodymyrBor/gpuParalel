import os

import numpy as np
import pycuda.autoinit
from pycuda import driver as drv
from pycuda.compiler import SourceModule

from matrix.common import measure_time, increment_by_one_cpu


if os.system("cl.exe"):
    os.environ['PATH'] += ';' + r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64"
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")

SIZE = 10


mod = SourceModule("""
__global__ void increment_by_one(float *array) {
    const int tx = threadIdx.x;
    //const int ty = blockIdx.x;
    //const int bw = blockDim.x;
    //const int index = tx + ty * bw;
    const int index = tx;
    array[index] = array[index] + 1;
}
""")


increment_by_one = mod.get_function('increment_by_one')

# an_array = np.random.randn(SIZE).astype(np.float32)
an_array = np.ones((SIZE, ))
an_array_copy = an_array.copy()
print(an_array)

print('[' + '.' * 30 + 'GPU' + '.' * 30 + ']')

for _ in range(1):
    with measure_time('GPU'):
        increment_by_one(
            drv.InOut(an_array),
            block=(SIZE, 1, 1),
            grid=(1, 1),
        )

print('[' + '.' * 30 + 'CPU' + '.' * 30 + ']')
for _ in range(3):
    with measure_time('CPU'):
        increment_by_one_cpu(an_array_copy)

print(f'GPU: {an_array}')
print(f'CPU: {an_array_copy}')
