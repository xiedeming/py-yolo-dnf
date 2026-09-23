# py-yolo-dnf

基于 Python + YOLO 的 PC 游戏自动通关程序。通过实时屏幕捕获、目标检测和决策状态机，实现游戏角色的自动刷图、释放技能、翻牌、捡装备、多角色轮换等全流程自动化。

> 本项目以 DNF（地下城与勇士：创新世纪）为目标游戏，架构上与具体游戏解耦，更换游戏只需重新训练检测模型并调整配置。

## 功能特性

- **实时目标检测** — Ultralytics YOLO（`.pt` / `.onnx` / TensorRT `.engine`）识别角色、怪物、门、道具、翻牌界面等 11 类目标
- **高性能屏幕捕获** — DXGI 桌面复制（bettercam）实测比 mss 快约 2.7 倍，全屏可达 41.5 FPS；初始化失败自动回退 mss
- **状态机决策** — 菜单 / 战斗 / 探索 / 过图 / 翻牌 / 卡住恢复 / Buff / 角色切换等状态自动流转
- **横版战斗逻辑** — 按目标类别智能放技能（怪物 1 个、小 Boss 2 个），技能冷却队列轮转
- **多角色轮换** — 每角色独立刷图次数、技能表、Buff 序列和移动速度系数
- **卡住自恢复** — 帧间差检测卡住，分级恢复（反向脱离 → 换向绕行 → 跳跃 → 释放重试）
- **防检测输入** — 贝塞尔曲线鼠标轨迹 + 随机化按键延迟
- **OCR 辅助** — RapidOCR（ONNX，无 Paddle 依赖）识别商店等界面文字
- **CPU 低配模式** — 无需 PyTorch/CUDA，ONNX Runtime + 精简模型即可运行

## 性能实测

主屏 2560×1440，RTX 3080 Laptop，两种捕获后端各 60 帧 × 3 轮交错取平均：

| 捕获后端 | 区域 | 捕获 | 推理 | 帧率 |
|---------|------|------|------|------|
| mss | 全屏 | 28.7ms | 35.3ms | 15.6 FPS |
| **bettercam** | 全屏 | 5.2ms | 18.8ms | **41.5 FPS** |
| mss | 1600×900 | 14.6ms | 24.4ms | 25.6 FPS |
| **bettercam** | 1600×900 | 4.2ms | 18.5ms | **43.9 FPS** |

捕获后端是主要瓶颈。注意「推理」列是 ultralytics `predict()` 的总耗时（含 CPU 上的 letterbox 和后处理）；跨批次数字会随系统负载漂移，比较后端必须同批次交错测量。

## 环境要求

- Windows 10/11（依赖 Win32 API 和 DXGI）
- Python 3.10+
- GPU 模式：NVIDIA 显卡 ≥ 4 GiB 显存 + CUDA
- CPU 模式：无额外硬件要求

## 安装

```bash
git clone https://github.com/xiedeming/py-yolo-dnf.git
cd py-yolo-dnf

# GPU 高配模式（含 PyTorch + Ultralytics）
pip install -r requirements.txt

# 或 CPU 低配模式（不含 PyTorch，用 ONNX Runtime）
pip install -r requirements-cpu.txt
```

启动时自动探测显卡：显存 ≥ 4 GiB 使用 `config/settings.yaml`（CUDA），否则使用 `config/settings.cpu.yaml`（CPU + ONNX Runtime）。可用 `-c` 显式指定配置，或 `-d cuda` / `-d cpu` 覆盖设备。

## 快速开始

```bash
# 使用默认配置（自动选择高低配）
python main.py

# 指定窗口和模型
python main.py -w "地下城与勇士：创新世纪" -m models/best.pt

# 列出所有可见窗口（用于确认窗口标题）
python main.py --list-windows

# 列出可用角色 / 地图
python main.py --list-characters
python main.py --list-maps

# 指定角色和地图，跳过自动进副本
python main.py --character DevilMayCry --stage dungeon_1 --no-enter

# 禁用调试窗口
python main.py --no-debug
```

### 命令行参数

| 参数 | 说明 |
|-----|------|
| `-c, --config` | 配置文件路径（默认按显存自动选择） |
| `-w, --window` | 游戏窗口标题（支持部分匹配） |
| `-m, --model` | YOLO 模型文件路径 |
| `-d, --device` | 推理设备：`cuda` / `cpu` |
| `-f, --fps` | 目标帧率（正整数） |
| `-C, --character` | 指定角色 ID |
| `-s, --stage` | 指定地图 ID |
| `--list-windows` | 列出所有可见窗口 |
| `--list-characters` | 列出可用角色 |
| `--list-maps` | 列出可用地图 |
| `--no-enter` | 跳过自动进入副本流程 |
| `--no-debug` | 禁用调试窗口 |

### 热键

| 按键 | 功能 |
|-----|------|
| P | 暂停 / 恢复 |
| Q / ESC | 退出 |

## 模型

将模型文件放入 `models/` 并在配置中指定路径。`models/` 现有候选（**类别顺序必须保持一致**：`people, door, monster, brand, menu, article, purple_card, hero, elite, boss-n, boss-m`）：

| 文件 | 架构 | 说明 |
|-----|------|------|
| `best_yolo26m.pt` | YOLO26m | **当前最优**，mAP50-95 0.7397 |
| `best.pt` / `best_sy.pt` | YOLO11m | 2026-04 训练，mAP50-95 0.7256 |
| `best_dxc.pt` | YOLO11m | 2025-07 训练，mAP50-95 0.6642 |

CPU 模式需要从训练机导出的 ONNX 模型：`models/cpu/best_dxc_yolo11n_416_fp32.onnx`（由 `scripts/export_cpu_onnx.py` 生成，缺失时程序会提示并退出）。训练与导出流程见 [TRAINING_ON_OTHER_MACHINE.md](TRAINING_ON_OTHER_MACHINE.md)。

## 架构

```
屏幕捕获 → YOLO检测 → 更新上下文 → 状态机决策 → 执行动作
    ↑                                              ↓
    └──────────────── 循环继续 ────────────────────┘
```

```
├── src/
│   ├── core/            # 主引擎、副本运行器、定时调度
│   ├── capture/         # bettercam(DXGI) / mss 捕获、窗口管理（工厂自动回退）
│   ├── detection/       # YOLODetector / MultiModelDetector / ONNX 检测器
│   ├── control/         # 键鼠模拟（贝塞尔曲线、人性化延迟）
│   ├── decision/        # 状态机、游戏上下文、技能/Buff/导航/翻牌/卡住恢复等
│   ├── selection/       # 角色/地图选择
│   ├── debug/           # 可视化调试窗口
│   └── utils/           # 配置加载、硬件探测、日志
├── config/              # settings.yaml（高配）/ settings.cpu.yaml（低配）
├── scripts/             # 测试与工具脚本
├── tests/               # 回归测试（unittest，不依赖显卡和模型权重）
├── models/              # 模型文件（.engine/.onnx 不入库，可重建）
└── main.py              # 入口
```

检测类别按语义分组驱动决策：`enemies`（monster/hero/elite/boss/boss-m）、`doors`、`items`、`ui_elements`（brand/menu/shop）、`players`、`cards` 等。

## 配置

主配置文件 `config/settings.yaml`，关键字段：

```yaml
game:
  window_title: "地下城与勇士：创新世纪"   # 需改为实际窗口标题
  target_fps: 30

capture:
  method: "bettercam"    # bettercam(DXGI，推荐) / mss

detection:
  device: "cuda"         # cuda / cpu
  models:
    main:
      path: "models/best_yolo26m.pt"

dungeon:
  mode: "new_abyss"      # abyss / white_map / new_abyss
  max_runs: 16           # 每角色最大刷图次数

decision:
  combat:
    skill_count_by_class:   # 遇到某类目标一次放几个技能
      monster: 1
      boss-m: 2
```

多角色在 `multi_character.role_list` 中配置，每个角色可指定技能表（`art` / `art_time`）、Buff 序列、移动速度系数（`move_speed`）等。

## 工具脚本

```bash
# 截屏实时推理（支持 .pt / .onnx / .engine）
python scripts/live_infer.py
python scripts/live_infer.py -m models/best_yolo26m_fp16.engine --capture bettercam
python scripts/live_infer.py --region 480 270 1600 900 --show -n 200
python scripts/live_infer.py -n 1 --save logs/snap.jpg

# 交互式测试
python scripts/test_capture.py          # 屏幕捕获
python scripts/test_capture.py -w       # 窗口捕获
python scripts/test_detection.py models/best.pt    # YOLO检测（摄像头）
python scripts/test_detection.py models/best.pt -s # YOLO检测（屏幕）
python scripts/test_control.py          # 输入控制

# CPU 模型训练 / 导出 / 基准（在训练机上执行）
python scripts/train_cpu_model.py --data D:\datasets\dnf\dataset.yaml --device 0
python scripts/export_cpu_onnx.py --weights runs/.../best.pt --imgsz 416
python scripts/benchmark_cpu_onnx.py
```

## 测试

```bash
# 回归测试（标准 unittest，不需要显卡、模型权重，不发送键鼠输入）
python -m unittest discover -s tests -v

# 语法检查
python -m compileall -q main.py src tests
```

覆盖 CLI 覆盖、配置持久化、检测阈值、决策状态优先级、技能分发、按键释放、引擎状态更新、卡住恢复等。`scripts/test_*.py` 为交互式诊断脚本，需在桌面会话中单独运行。

## 扩展

添加自定义状态回调：

```python
from src.core.engine import GameEngine
from src.decision.game_context import GameState

engine = GameEngine(config=config)

def on_combat(context):
    target = context.get_nearest_enemy()
    if target:
        engine.controller.mouse_move(target.center[0], target.center[1])
        engine.controller.mouse_click()

engine.set_state_callback(GameState.COMBAT, on_combat)
engine.start()
```

- 修改状态转换规则：编辑 `src/decision/state_machine.py` 的 `create_game_state_machine()`
- 添加检测类别：修改 `src/core/engine.py` 中的 `CLASS_MAPPING`
- 支持新游戏的技能数量规则：`config.decision.combat.skill_count_by_class` 加一行即可

更多实现细节见 [项目功能与流程文档.md](项目功能与流程文档.md) 与 [CLAUDE.md](CLAUDE.md)。

## 免责声明

本项目仅供计算机视觉与自动化技术的**学习研究**。使用自动化程序违反游戏服务条款，可能导致账号封禁，风险自负。请勿用于商业用途或破坏他人游戏体验。
