"""Static consistency check (no torch needed): every `from <local module> import X`
must refer to a name that module really defines. Guards against the bug where
FedMUModel was deleted from models/multimodal_model.py but still imported in 8 files."""
import ast
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".git", ".github", "fedmu_env", "venv", ".venv", "__pycache__"}


def _py_files():
    for dp, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if f.endswith(".py"):
                yield Path(dp) / f


def _defined_names(tree):
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(n.name)
        elif isinstance(n, ast.Assign):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
    return names


def test_local_imports_resolve():
    trees = {p: ast.parse(p.read_text(encoding="utf-8")) for p in _py_files()}
    defs = {".".join(p.relative_to(ROOT).with_suffix("").parts): _defined_names(t) for p, t in trees.items()}
    broken = []
    for p, tree in trees.items():
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.level == 0 and n.module in defs:
                broken += [f"{p.relative_to(ROOT)}:{n.lineno} {n.module}.{a.name}"
                           for a in n.names if a.name != "*" and a.name not in defs[n.module]]
    assert not broken, "Broken local imports:\n" + "\n".join(broken)


def test_all_files_compile():
    for p in _py_files():
        compile(p.read_text(encoding="utf-8"), str(p), "exec")
