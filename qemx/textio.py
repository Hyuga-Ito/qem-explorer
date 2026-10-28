"""Text I/O contract of QEM Explorer (1.7.0; file names 1.8.0): UTF-8, explicitly, everywhere.

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

File *names* are a different matter from file *contents* (1.8.0). The operating system decides how a path is
encoded: Windows always uses UTF-16 file names (Python's file-system encoding is UTF-8 there whatever the code page),
so "データ/結果.json" works with UTF-8 mode on or off. On Linux and macOS the file-system encoding follows the locale;
under a legacy locale such as ``en_US.ISO-8859-1`` a Japanese or Greek file name cannot be represented at all, by any
program. Such a path raises :class:`PathEncodingError`, which names the file-system encoding and the remedy (a UTF-8
locale), instead of an anonymous ``UnicodeEncodeError``. The contents rule (UTF-8) is unaffected.

Binary files (ZIP, NPZ, images, ``tomllib.load``) are opened in binary mode and take no encoding.
"""
from __future__ import annotations

import json
import os
import sys

ENCODING = "utf-8"


class TextEncodingError(ValueError):
    """A text file that is not valid UTF-8 (raised with its path and the offending byte position)."""


class PathEncodingError(ValueError):
    """A file *name* that the operating system's file-name encoding cannot represent (a property of the locale on
    Linux/macOS, never of the file's contents)."""


def path_representable(path) -> bool:
    """Whether this interpreter's file-system encoding can represent ``path`` (always true on Windows)."""
    try:
        os.fsencode(os.fspath(path))
        return True
    except UnicodeEncodeError:
        return False


def _path_error(path, e: UnicodeEncodeError):
    """A clear PathEncodingError when ``e`` came from encoding the path itself, else None (e.g. lone surrogates in
    the text being written, which UTF-8 cannot encode either)."""
    p = os.fspath(path)
    if not isinstance(p, str) or e.object != p:
        return None
    return PathEncodingError(
        "the file name %r cannot be represented in this system's file-name encoding (%s; character %r). This is set "
        "by the operating-system locale, not by QEM Explorer or the file's contents: use a UTF-8 locale (for example "
        "LANG=C.UTF-8 or en_US.UTF-8) or a path with characters that encoding supports."
        % (p, sys.getfilesystemencoding(), e.object[e.start:e.end]))


def _open(path, mode, **kw):
    try:
        return open(path, mode, **kw)
    except UnicodeEncodeError as e:
        err = _path_error(path, e)
        if err is None:
            raise
        raise err from None


def _explain(path, e: UnicodeDecodeError) -> TextEncodingError:
    return TextEncodingError(
        "%s is not valid UTF-8 (byte %d: %s). QEM Explorer reads and writes text files as UTF-8; re-save the file "
        "as UTF-8 (in most editors: Save As → Encoding: UTF-8)." % (os.fspath(path), e.start, e.reason))


def read_text(path) -> str:
    """The whole text of a UTF-8 file (strict decoding; a byte-order mark is kept as U+FEFF)."""
    try:
        with _open(path, "r", encoding=ENCODING) as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise _explain(path, e) from None


def read_config_text(path) -> str:
    """The text of a user-edited configuration or document (YAML, JSON): UTF-8, with or without a byte-order mark
    (the one compatibility path; strict decoding otherwise)."""
    try:
        with _open(path, "r", encoding="utf-8-sig") as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise _explain(path, e) from None


def write_text(path, text: str, *, atomic: bool = False) -> None:
    """Write ``text`` as UTF-8 (``newline=""`` keeps the text's own line endings on every platform). ``atomic``:
    write a temporary file next to it and rename it into place."""
    path = os.fspath(path)
    tmp = path + ".tmp%d" % os.getpid() if atomic else path
    with _open(tmp, "w", encoding=ENCODING, newline="") as f:
        f.write(text)
    if atomic:
        os.replace(tmp, path)


NONFINITE_KEY = "$nonfinite"         # 1.8.0: {JSON pointer: "nan" | "inf" | "-inf"} of values written as null


class NonStrictJSON(ValueError):
    """A JSON text with NaN/Infinity tokens where strict JSON was required."""


def _ptr(parts) -> str:
    return "".join("/" + str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def strict(obj, default=None):
    """(strict object, {pointer: kind}) — every non-finite float replaced by None, its location and kind recorded
    (1.8.0; review finding F9). ``default`` converts unknown objects first (as json.dumps would)."""
    import math
    found = {}

    def walk(x, parts):
        if isinstance(x, float):
            if math.isnan(x):
                found[_ptr(parts)] = "nan"
                return None
            if math.isinf(x):
                found[_ptr(parts)] = "inf" if x > 0 else "-inf"
                return None
            return x
        if isinstance(x, dict):
            return {k: walk(v, parts + [k]) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [walk(v, parts + [i]) for i, v in enumerate(x)]
        if isinstance(x, (str, int, bool)) or x is None:
            return x
        if default is not None:
            return walk(default(x), parts)
        raise TypeError("Object of type %s is not JSON serializable" % type(x).__name__)
    return walk(obj, []), found


def dumps(obj, *, indent=1, default=None, sort_keys: bool = False) -> str:
    """Strict JSON text (RFC 8259: no NaN or Infinity tokens). Non-finite numbers are written as ``null`` and listed,
    with their kind (``nan``: undefined or unavailable; ``inf``/``-inf``: unbounded), under the reserved top-level key
    ``$nonfinite`` as JSON pointers — never replaced by an arbitrary finite number. ``loads`` restores them. A
    top-level array with non-finite entries is wrapped as {"$array": [...], "$nonfinite": ...}."""
    s, found = strict(obj, default)
    if found:
        if isinstance(s, dict):
            s = dict(s)
            s[NONFINITE_KEY] = found
        else:
            s = {"$array": s, NONFINITE_KEY: {"/$array" + k: v for k, v in found.items()}}
    return json.dumps(s, indent=indent, ensure_ascii=False, sort_keys=sort_keys, allow_nan=False)


def _restore(obj):
    if not isinstance(obj, dict) or NONFINITE_KEY not in obj:
        return obj
    marks = obj.pop(NONFINITE_KEY)
    val = {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}
    for ptr, kind in marks.items():
        parts = [p.replace("~1", "/").replace("~0", "~") for p in ptr.split("/")[1:]]
        cur = obj
        for p in parts[:-1]:
            cur = cur[int(p)] if isinstance(cur, list) else cur[p]
        last = parts[-1]
        if isinstance(cur, list):
            cur[int(last)] = val[kind]
        else:
            cur[last] = val[kind]
    if set(obj) == {"$array"}:
        return obj["$array"]
    return obj


def loads(text: str, *, strict_tokens: bool = False):
    """Parse JSON written by ``dumps`` (non-finite values restored from ``$nonfinite``) or by QEM Explorer ≤ 1.7.0
    (bare NaN/Infinity tokens, accepted for compatibility). ``strict_tokens``: refuse bare NaN/Infinity tokens
    (validation of current exports)."""
    def bad(tok):
        raise NonStrictJSON("non-standard JSON token %s (written before 1.8.0, or not by QEM Explorer)" % tok)
    obj = json.loads(text, parse_constant=bad) if strict_tokens else json.loads(text)
    return _restore(obj)


def read_json(path, *, strict_tokens: bool = False):
    try:
        return loads(read_config_text(path), strict_tokens=strict_tokens)
    except NonStrictJSON as e:
        raise NonStrictJSON("%s: %s" % (os.fspath(path), e)) from None


def write_json(path, obj, *, indent=1, atomic: bool = True, default=None, sort_keys: bool = False) -> None:
    """Strict JSON (``dumps``) as UTF-8 text with non-ASCII characters kept as they are."""
    write_text(path, dumps(obj, indent=indent, default=default, sort_keys=sort_keys), atomic=atomic)


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
