"""
移动控制模块 - 把像素距离换算成方向键按住时长，并按角色移动速度系数调整

放在输入适配层（src/control/）而不是决策层：这里唯一的职责是把"距离"变成"按键时长"，
属于适配器的工作。`input_controller.py` 保持为纯粹的按键包装，不引入像素语义。

两条重要约定：
- **不阻塞主循环**。定时按住（`step`）只记录到期时刻，由主循环每帧调用 `update()` 释放，
  没有任何 `time.sleep()` 参与移动节奏。时钟以 `now` 注入，测试可替换。
- **按键来自配置**（`side_scroller.move_left_key` / `move_right_key`），不再硬编码
  `'left'/'right'`。直接驱动 `key_down`/`key_up`，因此 `release_all_inputs()` 依然能释放
  这些按键（它按 `_pressed_keys` 释放，与 `_current_moving_direction` 无关）。
"""
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

MIN_HOLD = 0.02  # move_speed 下限，避免除零


@dataclass
class MoveSpeedModel:
    """像素距离 → 按住时长的换算模型。`move_speed` 为角色速度系数，1.0 为基准，>1 更快。"""

    press_sleep: float = 0.55          # 角色配置：参考距离对应的按住时长(秒)
    run_sleep: float = 0.075           # 角色配置：最小点按/步进时长(秒)
    move_speed: float = 1.0            # 角色速度系数
    reference_distance: float = 150.0  # press_sleep 对应的像素距离
    max_hold: float = 1.2              # 单次定时按住上限(秒)

    def raw_hold(self, distance_px: float) -> float:
        """未钳制、未按速度缩放的基准时长；用于判断"这一步是否太远"。"""
        if distance_px <= 0:
            return 0.0
        return self.press_sleep * (distance_px / self.reference_distance)

    def hold_duration(self, distance_px: float) -> float:
        """
        把像素距离换算成按住时长。

        先按 run_sleep/max_hold 钳制再除以速度系数，这样两个边界值始终以"基准角色"
        为单位表达，慢角色不会被压到一个荒谬的下限。
        """
        if distance_px <= 0:
            return 0.0
        base = min(max(self.raw_hold(distance_px), self.run_sleep), self.max_hold)
        return base / max(self.move_speed, MIN_HOLD)

    def needs_hold(self, distance_px: float) -> bool:
        """距离远到一次定时点按盖不住，需要持续按住逐帧重规划。"""
        return self.raw_hold(distance_px) >= self.max_hold

    def scale_threshold(self, base_px: float) -> float:
        """按速度缩放像素判定阈值。速度越快，死区越大，避免越过目标后来回抖动。"""
        return base_px * self.move_speed


class MovementController:
    """输入控制器的移动层：把"走到目标"变成按住或点按，且不阻塞主循环。"""

    def __init__(
        self,
        controller,
        model: Optional[MoveSpeedModel] = None,
        left_key: str = 'left',
        right_key: str = 'right',
        dash_gap: float = 0.05,
        now: Callable[[], float] = time.monotonic,
    ):
        """
        Args:
            controller: InputController 实例
            model: 速度模型
            left_key / right_key: 方向键，来自 side_scroller 配置
            dash_gap: 跑动双击宏两次按键之间的间隔(秒)；测试可传 0
            now: 时钟，注入以便测试
        """
        self._controller = controller
        self._model = model or MoveSpeedModel()
        self._keys = {'left': left_key, 'right': right_key}
        self._dash_gap = dash_gap
        self._now = now

        self._held: Optional[str] = None          # 当前按住的方向
        self._deadline: Optional[float] = None    # 定时点按的到期时刻；None = 持续按住

    # ---------- 速度模型 ----------

    @property
    def model(self) -> MoveSpeedModel:
        return self._model

    def set_speed(self, move_speed: float, press_sleep: Optional[float] = None,
                  run_sleep: Optional[float] = None) -> None:
        """按当前角色更新速度系数与基准时长。"""
        safe_speed = move_speed if move_speed and move_speed > 0 else 1.0
        updates = {'move_speed': safe_speed}
        if press_sleep is not None and press_sleep > 0:
            updates['press_sleep'] = press_sleep
        if run_sleep is not None and run_sleep > 0:
            updates['run_sleep'] = run_sleep
        self._model = replace(self._model, **updates)

    def hold_duration(self, distance_px: float) -> float:
        return self._model.hold_duration(distance_px)

    def scale_threshold(self, base_px: float) -> float:
        return self._model.scale_threshold(base_px)

    # ---------- 移动指令 ----------

    def hold(self, direction: str) -> None:
        """持续按住某方向（跑动）。方向不变时是空操作，不会重复触发双击宏。"""
        if direction not in self._keys:
            return
        if self._held == direction and self._deadline is None:
            return

        self._release()
        key = self._keys[direction]

        # 跑动 = 双击方向键后长按，与 InputController.start_moving(run=True) 的行为一致。
        # 只在方向改变时触发一次，不是每帧。dash_gap 只控制间隔，不决定是否双击
        # （测试传 0 即可零间隔地走完整段双击）。
        self._controller.key_down(key)
        if self._dash_gap:
            time.sleep(self._dash_gap)
        self._controller.key_up(key)
        if self._dash_gap:
            time.sleep(self._dash_gap)
        self._controller.key_down(key)

        self._held = direction
        self._deadline = None

    def step(self, direction: str, distance_px: float) -> bool:
        """
        定时点按（不跑动），时长由剩余距离换算。

        Returns:
            是否真的起步（距离过小或方向非法时为 False）
        """
        if direction not in self._keys:
            return False

        duration = self._model.hold_duration(distance_px)
        if duration <= 0:
            return False

        self._release()
        self._controller.key_down(self._keys[direction])
        self._held = direction
        self._deadline = self._now() + duration
        return True

    def approach(self, direction: str, distance_px: float) -> None:
        """远距离持续按住（逐帧重规划），近距离定时点按（不会越过剩余距离）。"""
        if self._model.needs_hold(distance_px):
            self.hold(direction)
        else:
            self.step(direction, distance_px)

    def stop(self) -> None:
        """释放方向键并清空状态（暂停/停止/换角色时调用）。"""
        self._release()

    def update(self) -> None:
        """每帧调用：定时点按到期就释放方向键。替代阻塞 sleep。"""
        if self._deadline is not None and self._now() >= self._deadline:
            self._release()

    # ---------- 状态 ----------

    def is_moving(self) -> bool:
        return self._held is not None

    def is_stepping(self) -> bool:
        return self._deadline is not None

    def get_direction(self) -> Optional[str]:
        return self._held

    def _release(self) -> None:
        if self._held is not None:
            self._controller.key_up(self._keys[self._held])
        self._held = None
        self._deadline = None
