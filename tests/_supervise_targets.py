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


def heartbeat(path, seconds):
    """Append a line to ``path`` every 0.1 s for ``seconds`` (1.9.1: lets a test see whether a child still runs
    without looking up process IDs, which can belong to another PID namespace)."""
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as f:
        f.write("pid %d\n" % os.getpid())
        f.flush()
        while time.time() - t0 < float(seconds):
            f.write("%.3f\n" % time.time())
            f.flush()
            time.sleep(0.1)
    return "done"
