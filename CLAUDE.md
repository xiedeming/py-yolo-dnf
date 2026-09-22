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

### 截屏实时推理

`scripts/live_infer.py` 捕获屏幕（或指定区域）逐帧送模型推理，输出检测结果与实时帧率。
复用 `MSSCapture` + `YOLODetector`，因此 `.pt` / `.onnx` / `.engine` 都能直接加载。

```bash
# 默认使用 models/best_yolo26m.pt（捕获后端由 config 决定）
python scripts/live_infer.py

# 使用 TensorRT 引擎（推理快约 1.4x）
python scripts/live_infer.py -m models/best_yolo26m_fp16.engine

# 显式指定捕获后端: mss 或 bettercam(DXGI)
python scripts/live_infer.py --capture bettercam

# 只截游戏区域、显示窗口、跑 200 帧后统计
python scripts/live_infer.py --region 480 270 1600 900 --show -n 200

# 保存一张标注截图
python scripts/live_infer.py -n 1 --save logs/snap.jpg
```

实测帧率（主屏 2560×1440，RTX 3080 Laptop，两种后端各 60 帧 × 3 轮交错取平均）：

| 捕获后端 | 区域 | 捕获 | 推理 | 帧率 |
|---------|------|------|------|------|
| mss | 全屏 | 28.7ms | 35.3ms | 15.6 FPS |
| **bettercam** | 全屏 | 5.2ms | 18.8ms | **41.5 FPS**（2.66x） |
| mss | 1600×900 | 14.6ms | 24.4ms | 25.6 FPS |
| **bettercam** | 1600×900 | 4.2ms | 18.5ms | **43.9 FPS** |

**捕获后端是主要瓶颈**，换 bettercam 后全屏从 15.6 提到 41.5 FPS。注意两点：
- 「推理」一列是 ultralytics `predict()` 的**总耗时**，含 CPU 上的 letterbox 缩放和后处理，
  不只是引擎前向（约 9.8ms）。mss 在 CPU 上的大块拷贝还会拖慢同一颗芯片上的 GPU 推理
  （35.3ms vs 18.8ms，3 轮交错稳定复现），所以换捕获后端是双重收益。
- **跨批次数字会随系统负载漂移**（同一配置曾测到 25.7 FPS）。要比较后端必须**同批次交错**测量。

用 bettercam 之后再叠加 `--region` 收益已不明显。

DXGI 只在画面变化时交付新帧，`BetterCamCapture` 会复用上一帧并在结束时提示复用次数；
若屏幕长时间完全静止，首次抓取最多等待 1s 后报错。

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
- `bettercam_capture.py`: 基于 DXGI 桌面复制的捕获，实测比 mss 快约 6 倍（推荐）
- `mss_capture.py`: 基于MSS的捕获，兼容性最好的回退方案
- `__init__.py`: `create_capture(method, monitor_index, target_fps)` 工厂，bettercam 初始化失败会自动回退 mss
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
- `skill_manager.py`: `SkillQueue`（技能冷却队列）/ `BuffManager`
- `behavior_tree/`: 行为树实现，用于复杂决策

### 战斗技能释放

`src/core/engine.py` 的 `_combat_action_side_scroller` → `_perform_attack` 在进入攻击范围后
按**目标类别**释放技能：

| 目标类别 | 一次释放技能数 |
|---------|--------------|
| `monster` | 1 |
| `boss-m` | 2 |
| 其他（hero/elite/boss/people/menu…） | 0，只普攻 |

规则由 `config.decision.combat.skill_count_by_class` 配置，直接加行即可支持其他类别。
实际能否放出取决于 `SkillQueue` 的冷却：冷却中的技能会被轮转跳过，凑不够数量时放出
可用的那部分。`skill_trigger_interval`（默认 1.0s）限制触发频率，避免同一波敌人之间
每帧反复尝试；无论是否真的放出技能都会推进该节流时间。

回归测试：`tests/test_skill_trigger.py`。

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

`models/` 现有候选（类别顺序必须保持一致）:

| 文件 | 架构 | 说明 |
|-----|------|------|
| `best.pt` / `best_sy.pt` | YOLO11m | 2026-04 训练，内容相同 |
| `best_dxc.pt` | YOLO11m | 2025-07 训练 |
| `best_yolo26m.pt` | YOLO26m | **当前最优**，见下 |
| `best_yolo26m.onnx` | — | 同上的 ONNX 导出（opset 18，动态 shape） |
| `best_yolo26m_fp16.engine` | — | TensorRT FP16，静态 1×3×1280×1280 |

### 模型对比结论（固定验证集，139 训练 / 35 验证）

`best_yolo26m.pt` 由 COCO 预训练的 YOLO26m 在 174 张图上训练（imgsz 1280、batch 4、
100 epoch 上限 / patience 30），与两个 YOLO11m 微调版本使用完全相同的切分与超参：

| 模型 | mAP50-95 | mAP50 |
|-----|----------|-------|
| `best_yolo26m.pt` | **0.7397** | 0.9950 |
| `best_sy.pt` | 0.7256 | 0.9920 |
| `best_dxc.pt` | 0.6642 | 0.9559 |

验证集只有 62 个标注框，yolo26m 与 best_sy 的差距（0.014）在噪声范围内；能明确的是
best_dxc 更差。若要更确定的结论需扩充验证集。

注意：`best_yolo26m_fp16.engine` 体积 608 MB 且**与 GPU/驱动绑定**，换机器或升级驱动后
必须重新构建（`python main.py` 无法重建，需用训练环境里的 ultralytics 导出）。
