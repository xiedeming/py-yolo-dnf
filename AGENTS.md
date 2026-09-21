# Repository Guidelines

## Project Structure & Module Organization

This Windows Python application automates gameplay using screen capture, YOLOv8 detection, OCR, and keyboard/mouse control.

- `main.py`: CLI entry point and configuration overrides.
- `src/core/`: engine lifecycle, dungeon execution, and scheduling.
- `src/capture/`, `src/detection/`, `src/control/`: capture, perception, and input adapters.
- `src/decision/`: game context, state transitions, navigation, and skill management.
- `src/selection/`, `src/debug/`, `src/utils/`: selection, visualization, configuration, and logging.
- `config/settings.yaml`: runtime settings; `models/`: model assets.
- `scripts/test_*.py`: interactive diagnostics; `logs/`: runtime output.

## Build, Test, and Development Commands

Run commands from the repository root in a Windows virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --list-windows
python main.py -c config/settings.yaml -d cpu
```

The last two commands discover window titles and run the application with CPU inference. Use `-w "Game Title"` and `-m models/best.pt` to override the target window and model. Supply an existing model file. There is no separate build step.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` modules/functions/variables, and `PascalCase` classes. Follow existing type hints, dataclasses, and docstrings. Keep YAML indentation at two spaces. Preserve UTF-8 text, including Chinese comments. No formatter, linter, or type-checker configuration is currently provided.

## Testing Guidelines

Existing tests are interactive diagnostics, with no configured automated framework or coverage threshold:

- `python scripts/test_capture.py -w`: inspect window capture.
- `python scripts/test_detection.py models/best.pt -s -d cpu`: inspect screen detections.
- `python scripts/test_control.py --keyboard`: exercise keyboard input in a disposable editor window.

Control diagnostics send real input. Run affected diagnostics deliberately and record observed results. Follow `test_*.py` naming for new checks; isolate decision logic from desktop dependencies where practical.

## Commit & Pull Request Guidelines

History contains only `first commit`, so no established commit convention exists. Use concise, imperative subjects describing the affected behavior. PRs should explain the change, configuration/model requirements, and validation performed. Link relevant issues and include screenshots for visualization changes. Exclude generated logs, screenshots, caches, and unrelated model binaries.

## Configuration Changes

Keep `config/settings.yaml` and `src/utils/config_loader.py` aligned when adding settings. Verify window titles, model paths, and CPU/CUDA selection before running automation.
