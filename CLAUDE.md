# CLAUDE.md

> 本文件为 AI 助手提供项目上下文和开发规范，请在每次任务结束时更新此文件。

## Project Overview

Python + YOLOv8 PC游戏自动通关程序。通过屏幕捕获、目标检测和决策逻辑实现游戏自动化。

## Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Main Program
```bash
# 使用默认配置
python main.py

# 指定窗口和模型
python main.py -w "窗口标题" -m models/best.pt

# 列出所有可见窗口
python main.py --list-windows

# 使用CPU推理
python main.py -d cpu
```

### Run Test Scripts
```bash
# 测试屏幕捕获
python scripts/test_capture.py

# 测试窗口捕获
python scripts/test_capture.py -w

# 测试YOLOv8检测（摄像头）
python scripts/test_detection.py models/best.pt

# 测试YOLOv8检测（屏幕）
python scripts/test_detection.py models/best.pt -s

# 测试输入控制
python scripts/test_control.py
```

## Architecture

### Data Flow
```
屏幕捕获 → YOLOv8检测 → 更新上下文 → 状态机决策 → 执行动作
    ↑                                              ↓
    └──────────────── 循环继续 ────────────────────┘
```

### Core Modules

**src/core/engine.py** - 主引擎，协调所有模块的生命周期和主循环

**src/capture/** - 屏幕捕获
- `mss_capture.py`: 基于MSS的高性能屏幕捕获
- `window_manager.py`: Windows窗口定位和管理

**src/detection/detector.py** - YOLOv8推理
- `YOLODetector`: 单模型检测器
- `MultiModelDetector`: 多模型协同检测
- `Detection`: 检测结果数据类

**src/control/input_controller.py** - 键盘鼠标模拟
- 支持人性化延迟（防检测）
- 平滑鼠标移动（贝塞尔曲线）

**src/decision/** - 决策逻辑
- `game_context.py`: 游戏状态管理和历史记录
- `state_machine.py`: 有限状态机，定义状态转换规则
- `behavior_tree/`: 行为树实现，用于复杂决策

### Key Classes

```python
# 主引擎入口
from src.core.engine import GameEngine, EngineConfig

# 配置加载
from src.utils.config_loader import ConfigLoader, Config

# 状态枚举
from src.decision.game_context import GameState
```

### Configuration

配置文件: `config/settings.yaml`

关键字段:
- `game.window_title`: 游戏窗口标题（支持部分匹配）
- `detection.device`: `cuda` 或 `cpu`
- `detection.models`: 模型配置字典

### Extending Game Logic

添加自定义状态回调:
```python
engine = GameEngine(config=config)

def on_combat(context):
    target = context.get_nearest_enemy()
    if target:
        # 自定义攻击逻辑
        engine.controller.mouse_move(target.center[0], target.center[1])
        engine.controller.mouse_click()

engine.set_state_callback(GameState.COMBAT, on_combat)
engine.start()
```

修改状态转换规则: 编辑 `src/decision/state_machine.py` 中的 `create_game_state_machine()`

## Hotkeys

| 按键 | 功能 |
|-----|------|
| P | 暂停/恢复 |
| Q / ESC | 退出调试窗口 |

## Model Setup

将YOLOv8模型文件(.pt)放入 `models/` 目录，然后在配置文件中指定路径:

```yaml
detection:
  device: cuda
  models:
    main:
      path: "models/best.pt"
      conf_threshold: 0.5
```
