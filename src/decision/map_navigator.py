"""
地图导航模块 - 处理复杂门导航逻辑
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..control.input_controller import InputController
    from ..detection.detector import Detection
    from .game_context import GameContext


class Direction(Enum):
    """移动方向"""
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


@dataclass
class MapRoute:
    """地图路线配置"""
    map_id: str                           # 地图ID
    name: str = ""                        # 地图名称
    left_doors: List[int] = field(default_factory=list)   # 向左走的房间索引
    up_doors: List[int] = field(default_factory=list)     # 向上走的房间索引
    boss_room: int = 0                    # Boss房间索引
    total_rooms: int = 10                 # 总房间数

    def get_direction_for_room(self, room_index: int) -> Direction:
        """
        获取指定房间的移动方向

        Args:
            room_index: 房间索引

        Returns:
            移动方向
        """
        if room_index in self.left_doors:
            return Direction.LEFT
        elif room_index in self.up_doors:
            return Direction.UP
        else:
            return Direction.RIGHT  # 默认向右


@dataclass
class DoorInfo:
    """门信息"""
    position: Tuple[int, int]  # 门中心位置
    door_type: str = "normal"  # normal, left, up, boss
    is_boss_door: bool = False


class MapNavigator:
    """
    地图导航器

    功能：
    1. 管理多个地图的路线配置
    2. 根据当前房间索引决定移动方向
    3. 处理门检测结果，选择正确的门
    4. 支持Boss房间检测
    """

    def __init__(self, controller: 'InputController'):
        """
        初始化地图导航器

        Args:
            controller: 输入控制器
        """
        self.controller = controller

        # 路线配置字典 {map_id: MapRoute}
        self.routes: Dict[str, MapRoute] = {}

        # 当前激活的路线
        self.current_route: Optional[MapRoute] = None

        # 当前房间索引
        self.current_room: int = 0

    def add_route(self, route: MapRoute) -> None:
        """
        添加地图路线

        Args:
            route: 路线配置
        """
        self.routes[route.map_id] = route

    def set_route(self, map_id: str) -> bool:
        """
        设置当前激活的路线

        Args:
            map_id: 地图ID

        Returns:
            是否设置成功
        """
        if map_id in self.routes:
            self.current_route = self.routes[map_id]
            self.current_room = 0
            return True
        return False

    def get_door_direction(
        self,
        room_index: int,
        doors: List['Detection']
    ) -> Direction:
        """
        根据路线和门检测结果决定移动方向

        Args:
            room_index: 当前房间索引
            doors: 门检测结果列表

        Returns:
            移动方向
        """
        if self.current_route is None:
            # 无路线配置，默认向右
            return Direction.RIGHT

        return self.current_route.get_direction_for_room(room_index)

    def select_door(
        self,
        doors: List['Detection'],
        direction: Direction,
        screen_center: Tuple[int, int]
    ) -> Optional['Detection']:
        """
        从多个门中选择目标门

        Args:
            doors: 门检测结果列表
            direction: 目标方向
            screen_center: 屏幕中心坐标

        Returns:
            目标门Detection，如果没有合适的门则返回None
        """
        if not doors:
            return None

        if direction == Direction.RIGHT:
            # 选择右侧的门
            right_doors = [d for d in doors if d.center[0] > screen_center[0]]
            if right_doors:
                return min(right_doors, key=lambda d: d.center[0] - screen_center[0])
            # 没有右侧门，选择最右边的
            return max(doors, key=lambda d: d.center[0])

        elif direction == Direction.LEFT:
            # 选择左侧的门
            left_doors = [d for d in doors if d.center[0] < screen_center[0]]
            if left_doors:
                return max(left_doors, key=lambda d: d.center[0])
            # 没有左侧门，选择最左边的
            return min(doors, key=lambda d: d.center[0])

        elif direction == Direction.UP:
            # 选择上方的门
            up_doors = [d for d in doors if d.center[1] < screen_center[1]]
            if up_doors:
                return min(up_doors, key=lambda d: d.center[1])
            # 没有上方门，选择最上面的
            return min(doors, key=lambda d: d.center[1])

        elif direction == Direction.DOWN:
            # 选择下方的门
            down_doors = [d for d in doors if d.center[1] > screen_center[1]]
            if down_doors:
                return min(down_doors, key=lambda d: d.center[1] - screen_center[1])
            # 没有下方门，选择最下面的
            return max(doors, key=lambda d: d.center[1])

        # 默认返回第一个门
        return doors[0]

    def is_boss_room(self, room_index: int) -> bool:
        """
        判断当前房间是否为Boss房间

        Args:
            room_index: 房间索引

        Returns:
            是否为Boss房间
        """
        if self.current_route is None:
            return False

        return room_index >= self.current_route.boss_room

    def move_to_door(
        self,
        door: 'Detection',
        screen_center: Tuple[int, int],
        threshold: int = 50
    ) -> bool:
        """
        移动到门的位置

        Args:
            door: 目标门
            screen_center: 屏幕中心
            threshold: 停止阈值

        Returns:
            是否到达门
        """
        door_x = door.center[0]
        player_x = screen_center[0]

        # 判断方向
        if abs(door_x - player_x) <= threshold:
            # 已经在门附近
            return True

        if door_x < player_x:
            # 门在左边
            self.controller.key_down('left')
        else:
            # 门在右边
            self.controller.key_down('right')

        return False

    def enter_door(self) -> None:
        """进入门（向上）"""
        self.controller.key_press('up')

    def increment_room(self) -> None:
        """增加房间索引"""
        self.current_room += 1

    def reset_room(self) -> None:
        """重置房间索引"""
        self.current_room = 0

    def get_current_room(self) -> int:
        """获取当前房间索引"""
        return self.current_room

    def get_progress(self) -> float:
        """
        获取地图进度

        Returns:
            进度百分比 (0.0 - 1.0)
        """
        if self.current_route is None:
            return 0.0

        if self.current_route.total_rooms == 0:
            return 0.0

        return min(1.0, self.current_room / self.current_route.total_rooms)

    def get_status(self) -> dict:
        """
        获取当前状态

        Returns:
            状态字典
        """
        return {
            'current_map': self.current_route.map_id if self.current_route else None,
            'current_room': self.current_room,
            'total_rooms': self.current_route.total_rooms if self.current_route else 0,
            'progress': self.get_progress(),
            'is_boss_room': self.is_boss_room(self.current_room) if self.current_route else False
        }


def create_map_navigator_from_config(
    routes_config: Dict[str, any],
    controller: 'InputController'
) -> MapNavigator:
    """
    从配置创建地图导航器

    Args:
        routes_config: 路线配置字典（可以是dict或MapRouteConfig对象）
        controller: 输入控制器

    Returns:
        MapNavigator实例
    """
    navigator = MapNavigator(controller=controller)

    for map_id, route_data in routes_config.items():
        # 支持字典和MapRouteConfig对象两种类型
        if hasattr(route_data, 'left_doors'):
            # MapRouteConfig 对象
            route = MapRoute(
                map_id=map_id,
                name=getattr(route_data, 'name', ''),
                left_doors=list(route_data.left_doors) if route_data.left_doors else [],
                up_doors=list(route_data.up_doors) if route_data.up_doors else [],
                boss_room=route_data.boss_room if route_data.boss_room else 0,
                total_rooms=route_data.total_rooms if route_data.total_rooms else 10
            )
        else:
            # 字典
            route = MapRoute(
                map_id=map_id,
                name=route_data.get('name', ''),
                left_doors=route_data.get('left_doors', []),
                up_doors=route_data.get('up_doors', []),
                boss_room=route_data.get('boss_room', 0),
                total_rooms=route_data.get('total_rooms', 10)
            )
        navigator.add_route(route)

    return navigator
