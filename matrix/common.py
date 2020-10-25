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
