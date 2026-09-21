"""
配置加载模块
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict, dataclass, field


# ============ 技能和角色配置 ============

@dataclass
class SkillConfig:
    """技能配置"""
    id: str = ""
    name: str = ""
    key: str = ""
    cooldown: float = 10.0
    priority: int = 1
    condition_type: str = "always"
    condition_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackConfig:
    """普通攻击配置"""
    key: str = "space"
    type: str = "melee"  # melee / ranged
    range: int = 100


@dataclass
class CharacterSelectionConfig:
    """角色选择配置"""
    icon_position: Tuple[int, int] = (0, 0)
    icon_color: Tuple[int, int, int] = (0, 0, 0)
    confirm_key: str = "enter"


@dataclass
class CharacterConfig:
    """角色配置"""
    id: str = ""
    name: str = "Unknown"
    description: str = ""
    selection: CharacterSelectionConfig = None
    skills: List[SkillConfig] = field(default_factory=list)
    attack: AttackConfig = None

    def __post_init__(self):
        if self.selection is None:
            self.selection = CharacterSelectionConfig()
        if self.attack is None:
            self.attack = AttackConfig()


@dataclass
class CharactersConfig:
    """角色系统配置"""
    current: str = ""
    presets: Dict[str, CharacterConfig] = field(default_factory=dict)


# ============ 地图配置 ============

@dataclass
class MapSelectionConfig:
    """地图选择配置"""
    menu_key: str = "m"
    icon_position: Tuple[int, int] = (0, 0)
    scroll_count: int = 0
    confirm_key: str = "enter"


@dataclass
class MapFeaturesConfig:
    """地图特性配置"""
    has_boss: bool = False
    enemy_density: str = "medium"  # low/medium/high
    item_density: str = "medium"


@dataclass
class MapConfig:
    """地图配置"""
    id: str = ""
    name: str = "Unknown"
    description: str = ""
    selection: MapSelectionConfig = None
    features: MapFeaturesConfig = None

    def __post_init__(self):
        if self.selection is None:
            self.selection = MapSelectionConfig()
        if self.features is None:
            self.features = MapFeaturesConfig()


@dataclass
class MapsConfig:
    """地图系统配置"""
    current: str = ""
    presets: Dict[str, MapConfig] = field(default_factory=dict)


# ============ 横版游戏配置 ============

@dataclass
class SideScrollerConfig:
    """横版动作游戏配置"""
    move_left_key: str = "left"
    move_right_key: str = "right"
    attack_key: str = "x"
    attack_range: int = 100
    approach_threshold: int = 50
    attack_interval: float = 0.15


# ============ DNF 专用配置 ============

@dataclass
class BuffSkillConfig:
    """Buff技能配置"""
    keys: List[str] = field(default_factory=list)
    cooldown: float = 30.0
    apply_on_start: bool = True
    name: str = ""


@dataclass
class DungeonConfig:
    """副本配置"""
    mode: str = "white_map"           # abyss, white_map, new_abyss
    max_runs: int = 16                # 最大刷图次数
    auto_sell: bool = False           # 自动卖装备
    sell_after_runs: int = 5          # 刷图多少次后卖装备
    collect_items: bool = True        # 自动拾取
    menu_detect_threshold: int = 180  # menu连续检测次数阈值
    menu_timeout_seconds: Optional[float] = None
    character_button_pos: Tuple[int, int] = (960, 540)  # 选择角色按钮位置


@dataclass
class MapRouteConfig:
    """地图路线配置"""
    left_doors: List[int] = field(default_factory=list)   # 向左走的房间
    up_doors: List[int] = field(default_factory=list)     # 向上走的房间
    boss_room: int = 0                # Boss房间索引
    total_rooms: int = 10             # 总房间数


@dataclass
class CardFlipConfig:
    """翻牌配置"""
    enabled: bool = True
    purple_priority: bool = True      # 紫卡优先
    random_selection: bool = True     # 无紫卡时随机选择


@dataclass
class StuckRecoveryConfig:
    """卡住恢复配置"""
    door_threshold: int = 5           # 卡门阈值
    player_threshold: int = 4         # 玩家卡住阈值
    frame_similarity_threshold: float = 0.95  # 帧相似度阈值
    recovery_cooldown: float = 5.0    # 恢复动作冷却


@dataclass
class OCRConfig:
    """OCR配置 (基于RapidOCR/ONNX Runtime)"""
    enabled: bool = True              # 是否启用OCR
    # ONNX模型路径配置（可选，留空使用内置模型）
    det_model_dir: str = ""           # 文本检测模型路径
    rec_model_dir: str = ""           # 文本识别模型路径
    cls_model_dir: str = ""           # 文本方向分类模型路径


@dataclass
class ScheduleConfig:
    """定时调度配置"""
    enabled: bool = False
    hour: int = 6
    minute: int = 0
    recurring: bool = True            # 是否每日重复


@dataclass
class CharacterRunConfigData:
    """角色运行配置"""
    id: str = ""
    name: str = ""
    dungeon_runs: int = 16
    art: List[Any] = field(default_factory=list)  # 技能列表，支持单键和组合键
    art_time: Dict[str, float] = field(default_factory=dict)  # 技能冷却时间
    buff: List[List[str]] = field(default_factory=list)  # buff技能列表
    position: Tuple[int, int] = (0, 0)  # 保留兼容性


@dataclass
class MultiCharacterConfig:
    """多角色配置"""
    enabled: bool = False
    start_name: str = ""              # 开始角色名称
    end_name: str = ""                # 结束角色名称
    role_list: List[CharacterRunConfigData] = field(default_factory=list)


# ============ 决策配置 ============

@dataclass
class DecisionCombatConfig:
    """战斗决策配置"""
    attack_key: str = "space"
    target_priority: str = "nearest"  # nearest/weakest/strongest
    skill_strategy: str = "priority"  # priority/cooldown/custom
    use_skills: bool = True


@dataclass
class DecisionItemsConfig:
    """道具决策配置"""
    collect_key: str = "e"
    auto_collect: bool = True


@dataclass
class DecisionConfig:
    """决策配置"""
    states: List[str] = field(default_factory=lambda: [
        "menu", "loading", "playing", "combat", "paused", "dead", "victory"
    ])
    combat: DecisionCombatConfig = None
    items: DecisionItemsConfig = None

    def __post_init__(self):
        if self.combat is None:
            self.combat = DecisionCombatConfig()
        if self.items is None:
            self.items = DecisionItemsConfig()


# ============ 基础配置 ============

@dataclass
class GameConfig:
    """游戏配置"""
    window_title: str = "Game Window"
    target_fps: int = 30


@dataclass
class CaptureConfig:
    """捕获配置"""
    method: str = "mss"
    monitor_index: int = 1


@dataclass
class DetectionConfig:
    """检测配置"""
    device: str = "cuda"
    backend: str = "ultralytics"
    cpu_threads: int = 0
    models: Dict[str, dict] = None

    def __post_init__(self):
        if self.models is None:
            self.models = {}


@dataclass
class ControlConfig:
    """控制配置"""
    humanize: bool = True
    mouse_sensitivity: float = 1.0
    random_delay_range: tuple = (0.02, 0.08)


@dataclass
class DebugConfig:
    """调试配置"""
    enabled: bool = True
    show_detections: bool = True
    show_fps: bool = True
    show_state: bool = True
    save_screenshots: bool = False
    screenshot_interval: int = 100
    screenshot_dir: str = "data/screenshots"


@dataclass
class HotkeyConfig:
    """热键配置"""
    start: str = "f1"
    pause: str = "f2"
    stop: str = "f3"


@dataclass
class Config:
    """主配置"""
    game: GameConfig = None
    capture: CaptureConfig = None
    detection: DetectionConfig = None
    control: ControlConfig = None
    debug: DebugConfig = None
    hotkeys: HotkeyConfig = None
    decision: DecisionConfig = None
    characters: CharactersConfig = None
    maps: MapsConfig = None
    side_scroller: SideScrollerConfig = None
    # DNF 专用配置
    dungeon: DungeonConfig = None
    map_routes: Dict[str, MapRouteConfig] = field(default_factory=dict)
    card_flip: CardFlipConfig = None
    stuck_recovery: StuckRecoveryConfig = None
    schedule: ScheduleConfig = None
    multi_character: MultiCharacterConfig = None
    ocr: OCRConfig = None

    def __post_init__(self):
        if self.game is None:
            self.game = GameConfig()
        if self.capture is None:
            self.capture = CaptureConfig()
        if self.detection is None:
            self.detection = DetectionConfig()
        if self.control is None:
            self.control = ControlConfig()
        if self.debug is None:
            self.debug = DebugConfig()
        if self.hotkeys is None:
            self.hotkeys = HotkeyConfig()
        if self.decision is None:
            self.decision = DecisionConfig()
        if self.characters is None:
            self.characters = CharactersConfig()
        if self.maps is None:
            self.maps = MapsConfig()
        if self.side_scroller is None:
            self.side_scroller = SideScrollerConfig()
        # DNF 专用配置初始化
        if self.dungeon is None:
            self.dungeon = DungeonConfig()
        if self.card_flip is None:
            self.card_flip = CardFlipConfig()
        if self.stuck_recovery is None:
            self.stuck_recovery = StuckRecoveryConfig()
        if self.schedule is None:
            self.schedule = ScheduleConfig()
        if self.multi_character is None:
            self.multi_character = MultiCharacterConfig()
        if self.ocr is None:
            self.ocr = OCRConfig()


class ConfigLoader:
    """配置加载器"""

    @staticmethod
    def load(config_path: str = "config/settings.yaml") -> Config:
        """Load a YAML config, resolving an optional relative ``extends`` parent."""
        path = Path(config_path)
        if not path.exists():
            print(f"Config file not found: {config_path}, using defaults")
            return Config()
        return ConfigLoader._parse_config(ConfigLoader._load_yaml_tree(path, set()))


    @staticmethod
    def _load_yaml_tree(path: Path, visited: set) -> Dict[str, Any]:
        resolved = path.resolve()
        if resolved in visited:
            raise ValueError(f"Configuration extends cycle detected at: {path}")
        visited.add(resolved)
        try:
            with path.open('r', encoding='utf-8') as stream:
                data = yaml.safe_load(stream)
            if data is None:
                data = {}
            if not isinstance(data, dict):
                raise ValueError('Configuration root must be a mapping')

            parent = data.pop('extends', None)
            if parent is None:
                return data
            if not isinstance(parent, str) or not parent:
                raise ValueError('Configuration extends must be a non-empty path string')
            parent_path = Path(parent)
            if not parent_path.is_absolute():
                parent_path = path.parent / parent_path
            if not parent_path.is_file():
                raise FileNotFoundError(f"Parent configuration not found: {parent_path}")
            return ConfigLoader._merge_config_data(
                ConfigLoader._load_yaml_tree(parent_path, visited), data
            )
        finally:
            visited.remove(resolved)

    @staticmethod
    def _merge_config_data(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ConfigLoader._merge_config_data(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _parse_config(data: Dict[str, Any]) -> Config:
        """解析配置字典"""
        config = Config()

        if 'game' in data:
            game_data = data['game']
            config.game = GameConfig(
                window_title=game_data.get('window_title', 'Game Window'),
                target_fps=game_data.get('target_fps', 30)
            )

        if 'capture' in data:
            capture_data = data['capture']
            config.capture = CaptureConfig(
                method=capture_data.get('method', 'mss'),
                monitor_index=capture_data.get('monitor_index', 1)
            )

        if 'detection' in data:
            detection_data = data['detection']
            config.detection = DetectionConfig(
                device=detection_data.get('device', 'cuda'),
                backend=detection_data.get('backend', 'ultralytics'),
                cpu_threads=detection_data.get('cpu_threads', 0),
                models=detection_data.get('models', {})
            )

        if 'control' in data:
            control_data = data['control']
            delay_range = control_data.get('random_delay_range', [0.02, 0.08])
            config.control = ControlConfig(
                humanize=control_data.get('humanize', True),
                mouse_sensitivity=control_data.get('mouse_sensitivity', 1.0),
                random_delay_range=tuple(delay_range) if isinstance(delay_range, list) else delay_range
            )

        if 'debug' in data:
            debug_data = data['debug']
            config.debug = DebugConfig(
                enabled=debug_data.get('enabled', True),
                show_detections=debug_data.get('show_detections', True),
                show_fps=debug_data.get('show_fps', True),
                show_state=debug_data.get('show_state', True),
                save_screenshots=debug_data.get('save_screenshots', False),
                screenshot_interval=debug_data.get('screenshot_interval', 100),
                screenshot_dir=debug_data.get('screenshot_dir', 'data/screenshots')
            )

        if 'hotkeys' in data:
            hotkey_data = data['hotkeys']
            config.hotkeys = HotkeyConfig(
                start=hotkey_data.get('start', 'f1'),
                pause=hotkey_data.get('pause', 'f2'),
                stop=hotkey_data.get('stop', 'f3')
            )

        # 解析决策配置
        if 'decision' in data:
            config.decision = ConfigLoader._parse_decision_config(data['decision'])

        # 解析横版游戏配置
        if 'side_scroller' in data:
            ss_data = data['side_scroller']
            config.side_scroller = SideScrollerConfig(
                move_left_key=ss_data.get('move_left_key', 'left'),
                move_right_key=ss_data.get('move_right_key', 'right'),
                attack_key=ss_data.get('attack_key', 'x'),
                attack_range=ss_data.get('attack_range', 100),
                approach_threshold=ss_data.get('approach_threshold', 50),
                attack_interval=ss_data.get('attack_interval', 0.15)
            )

        # 解析角色配置
        if 'characters' in data:
            config.characters = ConfigLoader._parse_characters_config(data['characters'])

        # 解析地图配置
        if 'maps' in data:
            config.maps = ConfigLoader._parse_maps_config(data['maps'])

        # 解析DNF专用配置
        if 'dungeon' in data:
            config.dungeon = ConfigLoader._parse_dungeon_config(data['dungeon'])

        if 'map_routes' in data:
            config.map_routes = ConfigLoader._parse_map_routes_config(data['map_routes'])

        if 'card_flip' in data:
            config.card_flip = ConfigLoader._parse_card_flip_config(data['card_flip'])

        if 'stuck_recovery' in data:
            config.stuck_recovery = ConfigLoader._parse_stuck_recovery_config(data['stuck_recovery'])

        if 'schedule' in data:
            config.schedule = ConfigLoader._parse_schedule_config(data['schedule'])

        if 'multi_character' in data:
            config.multi_character = ConfigLoader._parse_multi_character_config(data['multi_character'])

        if 'ocr' in data:
            config.ocr = ConfigLoader._parse_ocr_config(data['ocr'])

        return config

    @staticmethod
    def _parse_decision_config(data: Dict) -> DecisionConfig:
        """解析决策配置"""
        decision = DecisionConfig()

        if 'states' in data:
            decision.states = data['states']

        if 'combat' in data:
            combat_data = data['combat']
            decision.combat = DecisionCombatConfig(
                attack_key=combat_data.get('attack_key', 'space'),
                target_priority=combat_data.get('target_priority', 'nearest'),
                skill_strategy=combat_data.get('skill_strategy', 'priority'),
                use_skills=combat_data.get('use_skills', True)
            )

        if 'items' in data:
            items_data = data['items']
            decision.items = DecisionItemsConfig(
                collect_key=items_data.get('collect_key', 'e'),
                auto_collect=items_data.get('auto_collect', True)
            )

        return decision

    @staticmethod
    def _parse_characters_config(data: Dict) -> CharactersConfig:
        """解析角色配置"""
        characters = CharactersConfig()
        characters.current = data.get('current', '')

        if 'presets' in data:
            for char_id, char_data in data['presets'].items():
                characters.presets[char_id] = ConfigLoader._parse_character_config(char_id, char_data)

        return characters

    @staticmethod
    def _parse_character_config(char_id: str, data: Dict) -> CharacterConfig:
        """解析单个角色配置"""
        # 解析选择配置
        selection_data = data.get('selection', {})
        selection = CharacterSelectionConfig(
            icon_position=tuple(selection_data.get('icon_position', [0, 0])),
            icon_color=tuple(selection_data.get('icon_color', [0, 0, 0])),
            confirm_key=selection_data.get('confirm_key', 'enter')
        )

        # 解析技能列表
        skills = []
        for skill_data in data.get('skills', []):
            skills.append(SkillConfig(
                id=skill_data.get('id', ''),
                name=skill_data.get('name', ''),
                key=skill_data.get('key', ''),
                cooldown=skill_data.get('cooldown', 10.0),
                priority=skill_data.get('priority', 1),
                condition_type=skill_data.get('condition', 'always'),
                condition_params=skill_data.get('condition_params', {})
            ))

        # 解析攻击配置
        attack_data = data.get('attack', {})
        attack = AttackConfig(
            key=attack_data.get('key', 'space'),
            type=attack_data.get('type', 'melee'),
            range=attack_data.get('range', 100)
        )

        return CharacterConfig(
            id=char_id,
            name=data.get('name', char_id),
            description=data.get('description', ''),
            selection=selection,
            skills=skills,
            attack=attack
        )

    @staticmethod
    def _parse_maps_config(data: Dict) -> MapsConfig:
        """解析地图配置"""
        maps = MapsConfig()
        maps.current = data.get('current', '')

        if 'presets' in data:
            for map_id, map_data in data['presets'].items():
                maps.presets[map_id] = ConfigLoader._parse_map_config(map_id, map_data)

        return maps

    @staticmethod
    def _parse_map_config(map_id: str, data: Dict) -> MapConfig:
        """解析单个地图配置"""
        # 解析选择配置
        selection_data = data.get('selection', {})
        selection = MapSelectionConfig(
            menu_key=selection_data.get('menu_key', 'm'),
            icon_position=tuple(selection_data.get('icon_position', [0, 0])),
            scroll_count=selection_data.get('scroll_count', 0),
            confirm_key=selection_data.get('confirm_key', 'enter')
        )

        # 解析特性配置
        features_data = data.get('features', {})
        features = MapFeaturesConfig(
            has_boss=features_data.get('has_boss', False),
            enemy_density=features_data.get('enemy_density', 'medium'),
            item_density=features_data.get('item_density', 'medium')
        )

        return MapConfig(
            id=map_id,
            name=data.get('name', map_id),
            description=data.get('description', ''),
            selection=selection,
            features=features
        )

    @staticmethod
    def _parse_dungeon_config(data: Dict) -> DungeonConfig:
        """解析副本配置"""
        char_button_pos = data.get('character_button_pos', [960, 540])
        return DungeonConfig(
            mode=data.get('mode', 'white_map'),
            max_runs=data.get('max_runs', 16),
            auto_sell=data.get('auto_sell', False),
            sell_after_runs=data.get('sell_after_runs', 5),
            collect_items=data.get('collect_items', True),
            menu_detect_threshold=data.get('menu_detect_threshold', 180),
            menu_timeout_seconds=data.get('menu_timeout_seconds'),
            character_button_pos=tuple(char_button_pos) if isinstance(char_button_pos, list) else char_button_pos
        )

    @staticmethod
    def _parse_map_routes_config(data: Dict) -> Dict[str, MapRouteConfig]:
        """解析地图路线配置"""
        routes = {}
        for map_id, route_data in data.items():
            routes[map_id] = MapRouteConfig(
                left_doors=route_data.get('left_doors', []),
                up_doors=route_data.get('up_doors', []),
                boss_room=route_data.get('boss_room', 0),
                total_rooms=route_data.get('total_rooms', 10)
            )
        return routes

    @staticmethod
    def _parse_card_flip_config(data: Dict) -> CardFlipConfig:
        """解析翻牌配置"""
        return CardFlipConfig(
            enabled=data.get('enabled', True),
            purple_priority=data.get('purple_priority', True),
            random_selection=data.get('random_selection', True)
        )

    @staticmethod
    def _parse_stuck_recovery_config(data: Dict) -> StuckRecoveryConfig:
        """解析卡住恢复配置"""
        return StuckRecoveryConfig(
            door_threshold=data.get('door_threshold', 5),
            player_threshold=data.get('player_threshold', 4),
            frame_similarity_threshold=data.get('frame_similarity_threshold', 0.95),
            recovery_cooldown=data.get('recovery_cooldown', 5.0)
        )

    @staticmethod
    def _parse_schedule_config(data: Dict) -> ScheduleConfig:
        """解析定时调度配置"""
        return ScheduleConfig(
            enabled=data.get('enabled', False),
            hour=data.get('hour', 6),
            minute=data.get('minute', 0),
            recurring=data.get('recurring', True)
        )

    @staticmethod
    def _parse_multi_character_config(data: Dict) -> MultiCharacterConfig:
        """解析多角色配置"""
        role_list = []
        for role_data in data.get('role_list', []):
            # 解析技能列表 (art) - 支持单键和组合键
            art = []
            for art_item in role_data.get('art', []):
                if isinstance(art_item, str):
                    # 单键，转换为列表形式
                    art.append([art_item])
                elif isinstance(art_item, list):
                    # 组合键
                    art.append(art_item)

            # 解析技能冷却时间 (art_time)
            art_time = role_data.get('art_time', {})

            # 解析buff技能
            buff = []
            for buff_item in role_data.get('buff', []):
                if isinstance(buff_item, list):
                    buff.append(buff_item)

            role_list.append(CharacterRunConfigData(
                id=role_data.get('id', ''),
                name=role_data.get('name', ''),
                dungeon_runs=role_data.get('dungeon_runs', 16),
                art=art,
                art_time=art_time,
                buff=buff,
                position=tuple(role_data.get('position', [0, 0]))
            ))

        return MultiCharacterConfig(
            enabled=data.get('enabled', False),
            start_name=data.get('start_name', ''),
            end_name=data.get('end_name', ''),
            role_list=role_list
        )

    @staticmethod
    def _parse_ocr_config(data: Dict) -> OCRConfig:
        """解析OCR配置"""
        return OCRConfig(
            enabled=data.get('enabled', True),
            det_model_dir=data.get('det_model_dir', ''),
            rec_model_dir=data.get('rec_model_dir', ''),
            cls_model_dir=data.get('cls_model_dir', '')
        )

    @staticmethod
    def save(config: Config, config_path: str = "config/settings.yaml") -> None:
        """保存所有配置项，保留可由 load 重新读取的 YAML 格式。"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(config)
        # YAML 的技能条件字段名与 dataclass 字段名不同。
        for character in data['characters']['presets'].values():
            for skill in character['skills']:
                skill['condition'] = skill.pop('condition_type')

        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    @staticmethod
    def create_default() -> Config:
        """
        创建默认配置

        Returns:
            默认Config对象
        """
        return Config()
