"""Static check: every project-owned text-mode I/O call names its encoding (1.7.0).

Python's text I/O without an ``encoding`` argument uses the locale's preferred encoding: UTF-8 on most Linux and
macOS systems, but a legacy code page (cp1252, cp932, …) on Windows unless UTF-8 mode is on. A file written on one
machine then fails to decode on another (``UnicodeDecodeError``). The project's contract is explicit UTF-8 for every
text file it reads or writes (JSON, YAML, CSV, Markdown, logs, exported scripts) and for the text it exchanges with
subprocesses; binary I/O (ZIP, NPZ, images, ``tomllib.load``) takes no encoding.

The check walks the abstract syntax tree of every maintained Python file and reports, with file and line:

* ``open`` / ``io.open`` / ``codecs.open`` / ``Path.open`` in text mode (no ``b`` in a literal mode) without
  ``encoding=``; a mode that is not a literal is reported too (it cannot be checked statically);
* ``Path.read_text`` / ``Path.write_text`` without ``encoding=``;
* ``tempfile.NamedTemporaryFile`` / ``TemporaryFile`` / ``SpooledTemporaryFile`` in text mode without ``encoding=``;
* pandas ``read_csv`` / ``read_table`` / ``read_json`` / ``read_fwf`` and ``DataFrame.to_csv`` / ``to_json`` with a
  destination, without ``encoding=``;
* ``subprocess.run`` / ``check_output`` / ``Popen`` / ``call`` / ``check_call`` with ``text=True`` or
  ``universal_newlines=True`` but no ``encoding=``;
* ``logging.FileHandler`` and ``numpy.savetxt`` / ``loadtxt`` / ``genfromtxt`` without ``encoding=``;
* ``errors="ignore"`` or ``errors="replace"`` on any of these calls (decoding errors must not be hidden).

Usage::

    python scripts/check_text_io.py [paths ...]      # default: the maintained tree; exit status 1 on findings

The check is deliberately syntactic: a call that passes ``**kwargs`` is reported unless the dictionary is a literal
that contains ``encoding``. Third-party packages are never scanned.
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIRS = ("qemx", "ui", "pages", "scripts", "tests", "examples", "app.py", "docs")
SKIP_DIRS = {"__pycache__", ".git", ".venv", "venv", "build", "dist", "node_modules", "archive"}

PANDAS_READ = {"read_csv", "read_table", "read_json", "read_fwf"}
PANDAS_WRITE = {"to_csv", "to_json"}
SUBPROCESS = {"run", "check_output", "Popen", "call", "check_call"}
TEMPFILES = {"NamedTemporaryFile", "TemporaryFile", "SpooledTemporaryFile"}
NUMPY_TEXT = {"savetxt", "loadtxt", "genfromtxt"}
HIDING = {"ignore", "replace"}


def _kw(call: ast.Call, name: str):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    return None


def _has_kw(call: ast.Call, name: str) -> bool:
    if _kw(call, name) is not None:
        return True
    for k in call.keywords:                       # **{"encoding": ...} literal
        if k.arg is None and isinstance(k.value, ast.Dict):
            if any(isinstance(x, ast.Constant) and x.value == name for x in k.value.keys):
                return True
    return False


def _star_kwargs(call: ast.Call) -> bool:
    return any(k.arg is None and not isinstance(k.value, ast.Dict) for k in call.keywords)


def _const(node):
    return node.value if isinstance(node, ast.Constant) else None


def _func_name(call: ast.Call) -> tuple:
    """(qualifier, attribute) of the called function: ('', 'open') for open(...), ('io', 'open') for io.open(...),
    ('<expr>', 'read_text') for p.read_text(...)."""
    f = call.func
    if isinstance(f, ast.Name):
        return "", f.id
    if isinstance(f, ast.Attribute):
        v = f.value
        q = v.id if isinstance(v, ast.Name) else ("<expr>" if not isinstance(v, ast.Attribute) else v.attr)
        return q, f.attr
    return "?", ""


def _mode_of(call: ast.Call, pos: int):
    """The mode argument (keyword ``mode`` or positional index ``pos``): a string, None when absent, or the AST
    node when it is not a literal."""
    m = _kw(call, "mode")
    if m is None and len(call.args) > pos:
        m = call.args[pos]
    if m is None:
        return None
    c = _const(m)
    return c if isinstance(c, str) else m


class Finding(tuple):
    __slots__ = ()

    def __new__(cls, path, line, msg):
        return tuple.__new__(cls, (path, line, msg))

    def __str__(self):
        return "%s:%d: %s" % self


def check_call(node: ast.Call) -> list:
    """Messages for one call (its arguments only; calls nested inside it are checked on their own)."""
    out = []
    q, name = _func_name(node)
    enc = _has_kw(node, "encoding") or _star_kwargs(node)
    err = _const(_kw(node, "errors")) if _kw(node, "errors") is not None else None
    text_call = False
    if name == "open" and q in ("", "io", "codecs", "<expr>"):
        pos = 0 if q == "<expr>" else 1               # Path.open(mode) vs open(file, mode)
        mode = _mode_of(node, pos)
        if isinstance(mode, ast.AST):
            if not enc:
                out.append("%s with a non-literal mode and no encoding= (cannot be checked statically)"
                           % ("open" if not q else q + ".open"))
            return out
        if mode is not None and "b" in mode:
            if _has_kw(node, "encoding"):
                out.append("binary-mode open with an encoding= argument")
            return out
        text_call = True
        if not enc:
            out.append("text-mode %s without encoding=" % ("open" if not q else q + ".open"))
    elif name in ("read_text", "write_text") and q not in ("", "?", "TX", "textio"):   # qemx.textio: UTF-8 always
        text_call = True
        if not enc:
            out.append("%s without encoding=" % name)
    elif name in TEMPFILES and q in ("tempfile", ""):
        mode = _mode_of(node, 0)
        if mode is None or (isinstance(mode, str) and "b" in mode):
            return out                                # default "w+b": binary
        if isinstance(mode, ast.AST):
            if not enc:
                out.append("%s with a non-literal mode and no encoding=" % name)
            return out
        text_call = True
        if not enc:
            out.append("text-mode %s without encoding=" % name)
    elif name in PANDAS_READ and q in ("pd", "pandas"):
        text_call = True
        if not enc:
            out.append("pandas.%s without encoding=" % name)
    elif name in PANDAS_WRITE and q not in ("", "?", "json"):
        has_dest = bool(node.args) or _kw(node, "path_or_buf") is not None
        if has_dest:
            text_call = True
            if not enc:
                out.append(".%s(...) to a file without encoding=" % name)
    elif name in SUBPROCESS and q == "subprocess":
        if _const(_kw(node, "text")) is True or _const(_kw(node, "universal_newlines")) is True:
            text_call = True
            if not enc:
                out.append("subprocess.%s(text=True) without encoding=" % name)
    elif name == "FileHandler" and q in ("logging", ""):
        text_call = True
        if not enc:
            out.append("logging.FileHandler without encoding=")
    elif name in NUMPY_TEXT and q in ("np", "numpy"):
        text_call = True
        if not enc:
            out.append("numpy.%s without encoding=" % name)
    if text_call and isinstance(err, str) and err in HIDING:
        out.append("errors=%r hides decoding errors" % err)
    return out


def check_source(src: str, path: str = "<string>") -> list:
    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:                       # a file that does not parse is reported, not skipped
        return [Finding(path, e.lineno or 0, "syntax error: %s" % e.msg)]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            out += [Finding(path, node.lineno, m) for m in check_call(node)]
    return sorted(out, key=lambda f: (f[1], f[2]))


def iter_files(paths) -> list:
    out = []
    for p in paths:
        p = os.path.join(ROOT, p) if not os.path.isabs(p) else p
        if os.path.isfile(p) and p.endswith(".py"):
            out.append(p)
            continue
        for d, dirs, files in os.walk(p):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            out += [os.path.join(d, f) for f in sorted(files) if f.endswith(".py")]
    return out


def check_paths(paths=DEFAULT_DIRS) -> list:
    found = []
    for f in iter_files(paths):
        with open(f, encoding="utf-8") as fh:
            src = fh.read()
        found += check_source(src, os.path.relpath(f, ROOT))
    return found


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    found = check_paths(argv or DEFAULT_DIRS)
    for f in found:
        print(f)
    print("%d finding%s" % (len(found), "" if len(found) == 1 else "s"), file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
