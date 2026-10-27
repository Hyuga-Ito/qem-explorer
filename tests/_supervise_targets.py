"""Functions run in supervised child processes by tests/test_selection_v170.py (importable by the child)."""
import os
import time


def sleep(seconds):
    time.sleep(float(seconds))
    return "slept"


def hog(gb):
    import numpy as np
    blocks = []
    for _ in range(int(gb * 8)):
        blocks.append(np.ones(2 ** 27 // 8 * 8, dtype=np.uint8))       # 128 MiB each, touched
        time.sleep(0.05)
    return len(blocks)


def crash():
    os._exit(7)


def boom():
    raise RuntimeError("exploded on purpose")


def rlimit_now():
    try:
        import resource
        return resource.getrlimit(resource.RLIMIT_AS)
    except ImportError:
        return None
