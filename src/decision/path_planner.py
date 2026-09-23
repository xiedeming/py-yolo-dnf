"""
路径规划接口 —— 决定"清完一间后往哪个方向走"

地下城（白图）与深渊的关键差别就在这里：白图是一间一间推进、每间要找出口，
深渊没有房间网格。而"往哪走"这个决策依赖两个信号：

- `door`：模型检不出（原训练数据已丢失，现训练集里 door 标注数为 0）
- 小地图：现在有了视觉参考（`D:\dnfwk\0a42efd2-5de9-11f0-9038-58cdc9c702c6.png`，
  一幅 2880×1800 画布、实际游戏画面 1920×1200 的实拍截图），`MinimapPathPlanner`
  按它校准实现

目前提供两种实现：

- `RightwardPlanner`：默认向右。不依赖任何检测，线性地图可用。
- `MinimapPathPlanner`：朝小地图上 BOSS 的方向推进。检测玩家标记（青）与 BOSS
  标记（红），在位移更大的轴上朝 BOSS 走。**是方向性推进，不是沿走廊的完整图搜索**；
  对梯子型房间布局有效，走廊绕行时靠卡住检测兜底。
"""
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .game_context import GameContext

logger = logging.getLogger(__name__)


class PathPlanner:
    """路径规划器接口：返回下一个该走的方向（'left' / 'right' / 'up' / 'down'）。"""

    def next_direction(self, context: 'GameContext', room_index: int, image=None) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        """用于日志/诊断的简短说明。"""
        return type(self).__name__


class RightwardPlanner(PathPlanner):
    """
    一直向右推进。

    对出口在右侧的线性地图有效；出口在左侧或需要上下的地图会走错方向 ——
    那种情况需要小地图（`MinimapPathPlanner`）或静态房间图配置。
    """

    def next_direction(self, context: 'GameContext', room_index: int, image=None) -> str:
        return 'right'


class MinimapPathPlanner(PathPlanner):
    """
    朝小地图上 BOSS 的方向推进。

    从当前帧裁出小地图面板（右上角，区域按画布比例），用颜色阈值分别找：
    - 玩家房间标记（青色竖条）
    - BOSS 标记（红角）

    取两者相对位移，在**位移更大的轴**（水平/垂直）上朝 BOSS 移动。对梯子型房间布局
    有效；走廊绕行时可能暂时到不了一墙之隔的房间，但推进会停滞，由卡住检测兜底。

    区域与颜色是从一张 1920×1200 实拍校准的（见 CLAUDE.md「地下城流程」一节）。
    """

    # 默认：小地图在屏幕右上角。区域 = (x 比例, y 比例, 宽比例, 高比例)
    REGION = (0.900, 0.020, 0.100, 0.090)
    # 玩家标记：青色竖条。HSV 规格 = (H 区间列表, S (下,上), V (下,上))
    PLAYER_HSV = (((90, 130),), (150, 255), (150, 255))
    # BOSS 标记：红色，H 在 0 附近或 180 附近，取 OR
    BOSS_HSV = (((0, 12), (170, 180)), (120, 255), (120, 255))
    MIN_AREA = 4  # 连通块最小面积，过滤噪点

    def __init__(self, region=None):
        """
        Args:
            region: 覆盖小地图面板的区域 (x_ratio, y_ratio, w_ratio, h_ratio)；
                    None 用默认校准值
        """
        self.region = tuple(region) if region else self.REGION

    def describe(self) -> str:
        return f"MinimapPathPlanner(region=({','.join(f'{v:.3f}' for v in self.region)}))"

    def next_direction(self, context: 'GameContext', room_index: int, image=None) -> str:
        """返回下一步方向；找不到玩家或 BOSS 时保守地向右（与默认行为一致）。"""
        if image is None:
            return 'right'

        player = self._find_blob(image, self.PLAYER_HSV)
        boss = self._find_blob(image, self.BOSS_HSV)
        if player is None or boss is None:
            return 'right'

        dx = boss[0] - player[0]
        dy = boss[1] - player[1]
        if abs(dx) >= abs(dy):
            return 'right' if dx > 0 else 'left'
        return 'down' if dy > 0 else 'up'

    def _find_blob(self, image, hsv_spec) -> tuple:
        """在小地图区域内找给定 HSV 规格的最大连通块中心，返回 (x, y) 或 None。"""
        h, w = image.shape[:2]
        rx, ry, rw, rh = self.region
        x0, y0 = int(w * rx), int(h * ry)
        x1, y1 = min(w, int(w * (rx + rw))), min(h, int(h * (ry + rh)))
        if x1 - x0 < 10 or y1 - y0 < 5:
            return None

        minimap = image[y0:y1, x0:x1]
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        mask = self._mask_of(hsv, hsv_spec)

        n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return None
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[idx, cv2.CC_STAT_AREA] < self.MIN_AREA:
            return None
        return (x0 + int(cents[idx][0]), y0 + int(cents[idx][1]))

    @staticmethod
    def _mask_of(hsv, hsv_spec):
        """按 HSV 规格生成掩码。H 通道支持多个区间（如红色在 0 和 180 两端），取 OR。"""
        h_ranges, (s_lo, s_hi), (v_lo, v_hi) = hsv_spec
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo_h, hi_h in h_ranges:
            lower = np.array([lo_h, s_lo, v_lo], dtype=np.uint8)
            upper = np.array([hi_h, s_hi, v_hi], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        return mask


# 配置值 -> 实现。新增实现时在这里登记，保持配置解析简单。
PLANNERS = {
    'rightward': RightwardPlanner,
    'minimap': MinimapPathPlanner,
}


def create_path_planner(name: str = 'rightward', region=None) -> PathPlanner:
    """
    按配置名创建路径规划器。

    Args:
        name: 'rightward' 或 'minimap'
        region: 小地图区域（仅 minimap 使用）
    """
    key = (name or 'rightward').lower()
    if key not in PLANNERS:
        raise ValueError(f"未知的路径规划器: {name}，可选: {sorted(PLANNERS)}")
    planner = PLANNERS[key](region=region) if key == 'minimap' else PLANNERS[key]()
    logger.info(f"路径规划器: {planner.describe()}")
    return planner