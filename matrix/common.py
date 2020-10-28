import os
from datetime import datetime
from contextlib import contextmanager

import numpy as np


def increment_by_one_cpu(array: np.ndarray):
    for i, value in enumerate(array):
        array[i] = value + 1


@contextmanager
def measure_time(text: str):
    start = datetime.now()
    print(f'{text} start at: {start}')
    try:
        yield
    finally:
        finish = datetime.now()
        delta = finish - start
        print(f'{text} finish at: {finish}. Delta: {delta.microseconds}.')


def add_compiler_to_PATH():
    if os.system("cl.exe"):
        os.environ['PATH'] += ';' + r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64"
    if os.system("cl.exe"):
        raise RuntimeError("cl.exe still not found, path probably incorrect")
