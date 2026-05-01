"""
smoke_test.py
Run this after environment setup to confirm all libraries load correctly.
Expected output: all checks show OK.
"""

import sys

def check(name, fn):
    try:
        result = fn()
        print(f"  OK  {name}: {result}")
        return True
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        return False

print("\n" + "="*55)
print("  Fed-MU Environment Smoke Test")
print("="*55)

passed = 0
total = 0

checks = [
    ("Python version",
     lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),

    ("PyTorch",
     lambda: __import__('torch').__version__),

    ("Transformers",
     lambda: __import__('transformers').__version__),

    ("PEFT",
     lambda: __import__('peft').__version__),

    ("Flower (flwr)",
     lambda: __import__('flwr').__version__),

    ("scikit-learn",
     lambda: __import__('sklearn').__version__),

    ("NumPy",
     lambda: __import__('numpy').__version__),

    ("PIL (Pillow)",
     lambda: __import__('PIL').__version__),

    ("PyYAML",
     lambda: __import__('yaml').__version__),

    ("Tensor creation (CPU)",
     lambda: str(__import__('torch').zeros(3, 3).shape)),

    ("Config file readable",
     lambda: str(__import__('yaml').safe_load(
         open('configs/fedmu_config.yaml'))['project']['name'])),
]

for name, fn in checks:
    total += 1
    if check(name, fn):
        passed += 1

print("="*55)
print(f"  Result: {passed}/{total} checks passed")
if passed == total:
    print("  Environment is READY. Proceed to Step 2.")
else:
    print("  Fix failed checks before proceeding.")
print("="*55 + "\n")