"""Text I/O contract of QEM Explorer (1.7.0): UTF-8, explicitly, everywhere.

Python's text I/O without an ``encoding`` argument uses the locale's preferred encoding — UTF-8 on most Linux and
macOS systems, a legacy code page (cp1252, cp932, …) on Windows unless UTF-8 mode is on. A JSON record, YAML
configuration or CSV table written on one machine then fails to decode on another (``UnicodeDecodeError``), or a
command line that prints "⟨O⟩" to a pipe fails to encode. Every project-owned text file is therefore read and
written as UTF-8 with the encoding named in the call (``scripts/check_text_io.py`` enforces it statically), and
decoding errors are never hidden (no ``errors="ignore"`` or ``"replace"``).

Compatibility paths, explicit and documented:

* ``read_config_text`` accepts a UTF-8 byte-order mark (files saved by older Windows editors) by decoding with
  ``utf-8-sig``; any other non-UTF-8 input (a file saved in cp1252 or Shift-JIS) is refused with the file name and
  byte position, never guessed.
* ``utf8_stdio`` makes a command line's standard output and error UTF-8 (``reconfigure``), so its printed text has
  one declared encoding whether it goes to a console, a pipe or a file. Console output on Windows is unaffected:
  Python writes the console through its wide-character API either way.
* ``child_env`` gives a Python subprocess ``PYTHONIOENCODING=utf-8``, so that a parent reading its output with
  ``encoding="utf-8"`` and the child writing it agree on every platform.

Binary files (ZIP, NPZ, images, ``tomllib.load``) are opened in binary mode and take no encoding.
"""
from __future__ import annotations

import json
import os
import sys

ENCODING = "utf-8"


class TextEncodingError(ValueError):
    """A text file that is not valid UTF-8 (raised with its path and the offending byte position)."""


def _explain(path, e: UnicodeDecodeError) -> TextEncodingError:
    return TextEncodingError(
        "%s is not valid UTF-8 (byte %d: %s). QEM Explorer reads and writes text files as UTF-8; re-save the file "
        "as UTF-8 (in most editors: Save As → Encoding: UTF-8)." % (os.fspath(path), e.start, e.reason))


def read_text(path) -> str:
    """The whole text of a UTF-8 file (strict decoding; a byte-order mark is kept as U+FEFF)."""
    try:
        with open(path, encoding=ENCODING) as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise _explain(path, e) from None


def read_config_text(path) -> str:
    """The text of a user-edited configuration or document (YAML, JSON): UTF-8, with or without a byte-order mark
    (the one compatibility path; strict decoding otherwise)."""
    try:
        with open(path, encoding="utf-8-sig") as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise _explain(path, e) from None


def write_text(path, text: str, *, atomic: bool = False) -> None:
    """Write ``text`` as UTF-8 (``newline=""`` keeps the text's own line endings on every platform). ``atomic``:
    write a temporary file next to it and rename it into place."""
    path = os.fspath(path)
    tmp = path + ".tmp%d" % os.getpid() if atomic else path
    with open(tmp, "w", encoding=ENCODING, newline="") as f:
        f.write(text)
    if atomic:
        os.replace(tmp, path)


def read_json(path):
    return json.loads(read_config_text(path))


def write_json(path, obj, *, indent=1, atomic: bool = True, default=None, sort_keys: bool = False) -> None:
    """JSON as UTF-8 text with non-ASCII characters kept as they are (``ensure_ascii=False``)."""
    write_text(path, json.dumps(obj, indent=indent, default=default, ensure_ascii=False, sort_keys=sort_keys),
               atomic=atomic)


def read_yaml(path):
    import yaml
    return yaml.safe_load(read_config_text(path))


def utf8_stdio() -> None:
    """Make standard output and error UTF-8 (call at the start of a command line's ``main``)."""
    for name in ("stdout", "stderr"):
        s = getattr(sys, name, None)
        rec = getattr(s, "reconfigure", None)
        enc = (getattr(s, "encoding", "") or "").lower().replace("_", "-")
        if rec is not None and enc not in ("utf-8", "utf8"):
            try:
                rec(encoding=ENCODING)
            except (ValueError, OSError):              # a stream that cannot be reconfigured is left as it is
                pass


def child_env(env: dict | None = None) -> dict:
    """Environment for a Python subprocess whose text output is read as UTF-8."""
    out = dict(os.environ if env is None else env)
    out["PYTHONIOENCODING"] = ENCODING
    return out
