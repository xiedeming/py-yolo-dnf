# Regression Tests

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q main.py src tests
```

The suite uses Python's standard `unittest` framework and the project's NumPy
and PyYAML dependencies. Hardware adapters and model loading are replaced with
test doubles; these tests do not send keyboard/mouse input or load YOLO weights.

Coverage includes CLI overrides, configuration persistence, detector thresholds,
decision-state priority, skill dispatch, input release, and engine state updates.

The interactive diagnostics in `scripts/test_*.py` remain separate. They require
the Windows runtime dependencies and, for detection, a model file. Run them
deliberately in a suitable desktop session after the automated checks.
