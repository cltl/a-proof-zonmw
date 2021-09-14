from time import time
from functools import wraps


def timer(func):
    @wraps
    def wrapper(func, *args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Elapsed time: {end - start}s")
        return result
    return wrapper