"""Print the facts about a test machine that decide floating-point last bits (1.8.0, docs/CI_FAILURE_ANALYSIS.md).

GitHub-hosted runners with the same label run on different CPUs; NumPy dispatches different SIMD kernels on AVX-512
and AVX2 machines, and OpenBLAS picks its kernels by CPU. Results agree to rounding, not bitwise. CI logs did not
record the CPU, so a failure that depends on it could not be attributed from the log alone; every job now prints it.

    python scripts/ci_runner_info.py                      # human-readable summary
    python scripts/ci_runner_info.py --avx512-disable     # NPY_DISABLE_CPU_FEATURES value that turns off this
                                                          # machine's AVX-512 dispatch ("" if it has none)
"""
from __future__ import annotations

import platform
import sys


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def numpy_features() -> dict:
    import numpy as np
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as feats      # NumPy 2
    except ImportError:                                                         # pragma: no cover - NumPy 1.x
        from numpy.core._multiarray_umath import __cpu_features__ as feats
    return {k: bool(v) for k, v in feats.items()}, np.__version__


def dispatch_targets() -> list:
    """The SIMD targets this NumPy build dispatches to at run time (``found`` in ``numpy.show_config``)."""
    import numpy as np
    try:
        simd = np.show_config(mode="dicts").get("SIMD Extensions", {})
        return list(simd.get("found", []))
    except Exception:                                                           # pragma: no cover - old NumPy
        return []


def avx512_disable_list() -> str:
    feats, _ = numpy_features()
    names = [k for k in dispatch_targets() if feats.get(k) and (k.startswith("AVX512") or k == "X86_V4")]
    return " ".join(names)


def main(argv) -> int:
    if "--avx512-disable" in argv:
        print(avx512_disable_list())
        return 0
    feats, ver = numpy_features()
    on = [k for k, v in feats.items() if v]
    print("python   ", sys.version.split()[0], platform.platform())
    print("cpu      ", cpu_model())
    print("numpy    ", ver, "SIMD available:", " ".join(on) or "-")
    print("dispatch ", " ".join(k for k in dispatch_targets() if feats.get(k)) or "baseline only")
    try:
        from threadpoolctl import threadpool_info
        for lib in threadpool_info():
            print("blas     ", lib.get("internal_api"), lib.get("version"), "core:", lib.get("architecture"),
                  "threads:", lib.get("num_threads"))
    except Exception as e:                                                      # informational only
        print("blas      (threadpoolctl unavailable: %s)" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
