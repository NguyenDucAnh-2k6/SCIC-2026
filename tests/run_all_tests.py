"""
ClassTwin Complete Test Suite Runner.
Discovers and executes all test functions across tests/test_*.py.
"""
import sys
from pathlib import Path
import inspect
import traceback

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "tests"))

test_files = [
    "test_schemas",
    "test_twin",
    "test_cql",
    "test_state_estimation",
    "test_inference",
    "test_rollout_evolution"
]

passed = 0
failed = 0
errors = []

for mod_name in test_files:
    print(f"\n================ Running {mod_name} ================")
    try:
        mod = __import__(mod_name)
    except Exception as e:
        print(f"FAILED to import {mod_name}: {e}")
        failed += 1
        errors.append((mod_name, "import", traceback.format_exc()))
        continue

    for attr in dir(mod):
        if attr.startswith("test_") and inspect.isfunction(getattr(mod, attr)):
            func = getattr(mod, attr)
            try:
                func()
                print(f"  [PASS] {attr}")
                passed += 1
            except Exception as e:
                print(f"  [FAIL] {attr}: {e}")
                failed += 1
                errors.append((mod_name, attr, traceback.format_exc()))

print("\n" + "=" * 60)
print(f"TEST SUITE SUMMARY: {passed} PASSED | {failed} FAILED")
print("=" * 60)

if failed > 0:
    for mod, fn, tb in errors:
        print(f"\n--- Failure in {mod}.{fn} ---")
        print(tb)
    sys.exit(1)
else:
    print("ALL TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)

