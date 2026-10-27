"""Image rendering in a child process (``python -m qemx.render_worker JOBDIR``), used by ``qemx.bundle``.

The job directory holds ``job.json``: {"items": [[file stem, Plotly JSON], …], "formats": […], "scale": …}. Every
image is written as ``<stem>.<fmt>.part`` and renamed to ``<stem>.<fmt>`` when complete, so the parent can collect
finished images at any time — also after it has killed this process. ``done`` marks a complete run; ``error.txt``
records why rendering stopped. Nothing here reads anything but the job file the parent wrote.

Rendering lives in its own process because a renderer can fail in ways that never return to its caller: Kaleido's
persistent-server thread can die (e.g. no Chrome) while the caller waits on its queue forever (QA report §4.6). The
parent enforces the time limit and kills this process; the numerical export never waits on it.
"""
from __future__ import annotations

import json
import os
import sys


def _write(path: str, data: bytes) -> None:
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)


def _find_browser():
    """Path of a Chrome/Chromium Kaleido can use, False if we can tell there is none, None if we cannot tell (an
    older or newer choreographer without this API: then the first image decides)."""
    p = os.environ.get("BROWSER_PATH")
    if p:
        return p if os.path.exists(p) else False
    try:
        from choreographer.browsers.chromium import Chromium
        return Chromium.find_browser(skip_local=False) or False
    except Exception:                                    # noqa: BLE001
        return None


def main(jobdir: str) -> int:
    with open(os.path.join(jobdir, "job.json"), encoding="utf-8") as f:
        job = json.load(f)
    formats, scale = list(job["formats"]), float(job["scale"])
    try:
        import warnings
        import plotly.io as pio
        warnings.simplefilter("ignore")
        fault = os.environ.get("QEMX_RENDER_FAULT", "")      # fault injection for the export's own tests only
        if fault == "hang":
            import time
            time.sleep(3600)
        elif fault == "crash":
            os._exit(9)
        elif fault == "chrome":
            raise RuntimeError("Kaleido v1 and later requires Chrome to be installed (simulated)")
        figs = [(stem, pio.from_json(js)) for stem, js in job["items"]]
        server = False
        first = True
        browser = _find_browser()
        if browser is False:                             # we can tell that there is no browser: fail fast
            raise RuntimeError("no Chrome or Chromium browser was found (Kaleido ≥ 1 needs one to render images)")
        if browser and job.get("server", True):
            # a browser exists: one persistent browser for every image (about 4× faster than one per image)
            try:
                import kaleido
                kaleido.start_sync_server(silence_warnings=True)
                server, first = True, False
            except Exception:                            # noqa: BLE001 - render image by image instead
                server = False
        for fmt in formats:
            for stem, fig in figs:
                w, h = int(fig.layout.width or 1400), int(fig.layout.height or 600)
                data = pio.to_image(fig, format=fmt, width=w, height=h, scale=scale if fmt == "png" else 1)
                _write(os.path.join(jobdir, "%s.%s" % (stem, fmt)), data)
                if fault == "after-first":
                    os._exit(9)
                if first and job.get("server", True):
                    first = False
                    # the first image proved a working browser: one persistent browser for the rest (about 4×
                    # faster). If it cannot start, rendering continues image by image; if it dies later, the
                    # parent's time limit ends this process and keeps what was written.
                    try:
                        import kaleido
                        kaleido.start_sync_server(silence_warnings=True)
                        server = True
                    except Exception:                    # noqa: BLE001
                        server = False
        if server:
            try:
                import kaleido
                kaleido.stop_sync_server(silence_warnings=True)
            except Exception:                            # noqa: BLE001
                pass
        _write(os.path.join(jobdir, "done"), b"ok")
        return 0
    except BaseException as e:                           # noqa: BLE001 - recorded for the parent
        msg = "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:300] if str(e) else "")
        try:
            _write(os.path.join(jobdir, "error.txt"), msg.encode("utf-8"))
        except Exception:                                # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
