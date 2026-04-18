"""
行为树模块 - 复杂决策逻辑
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Callable, Any


class NodeStatus(Enum):
    """节点状态"""
    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class BehaviorNode(ABC):
    """行为树节点基类"""

    def __init__(self, name: str = ""):
        self.name = name
        self.parent: Optional['BehaviorNode'] = None

    @abstractmethod
    def tick(self, context: Any) -> NodeStatus:
        """执行节点"""
        pass

    def reset(self) -> None:
        """重置节点状态"""
        pass


class Sequence(BehaviorNode):
    """
    顺序节点 - 所有子节点成功才成功

    按顺序执行所有子节点，如果任一子节点失败则返回失败
    """

    def __init__(self, children: List[BehaviorNode], name: str = "Sequence"):
        super().__init__(name)
        self.children = children
        self.current_index = 0
        for child in children:
            child.parent = self

    def tick(self, context: Any) -> NodeStatus:
        while self.current_index < len(self.children):
            status = self.children[self.current_index].tick(context)

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.FAILURE:
                self.current_index = 0
                return NodeStatus.FAILURE

            self.current_index += 1

        self.current_index = 0
        return NodeStatus.SUCCESS

    def reset(self) -> None:
        self.current_index = 0
        for child in self.children:
            child.reset()


class Selector(BehaviorNode):
    """
    选择节点 - 任一子节点成功即成功

    按顺序执行子节点，如果任一子节点成功则返回成功
    """

    def __init__(self, children: List[BehaviorNode], name: str = "Selector"):
        super().__init__(name)
        self.children = children
        self.current_index = 0
        for child in children:
            child.parent = self

    def tick(self, context: Any) -> NodeStatus:
        while self.current_index < len(self.children):
            status = self.children[self.current_index].tick(context)

            if status == NodeStatus.RUNNING:
                return NodeStatus.RUNNING
            elif status == NodeStatus.SUCCESS:
                self.current_index = 0
                return NodeStatus.SUCCESS

            self.current_index += 1

        self.current_index = 0
        return NodeStatus.FAILURE

    def reset(self) -> None:
        self.current_index = 0
        for child in self.children:
            child.reset()


class Action(BehaviorNode):
    """动作节点 - 执行具体动作"""

    def __init__(
        self,
        action_func: Callable[[Any], NodeStatus],
        name: str = "Action"
    ):
        super().__init__(name)
        self.action_func = action_func

    def tick(self, context: Any) -> NodeStatus:
        try:
            return self.action_func(context)
        except Exception as e:
            print(f"Action error: {e}")
            return NodeStatus.FAILURE


class Condition(BehaviorNode):
    """条件节点 - 检查条件"""

    def __init__(
        self,
        condition_func: Callable[[Any], bool],
        name: str = "Condition"
    ):
        super().__init__(name)
        self.condition_func = condition_func

    def tick(self, context: Any) -> NodeStatus:
        try:
            return NodeStatus.SUCCESS if self.condition_func(context) else NodeStatus.FAILURE
        except Exception as e:
            print(f"Condition error: {e}")
            return NodeStatus.FAILURE


class Decorator(BehaviorNode):
    """装饰器节点基类 - 修改子节点行为"""

    def __init__(self, child: BehaviorNode, name: str = "Decorator"):
        super().__init__(name)
        self.child = child
        child.parent = self

    def tick(self, context: Any) -> NodeStatus:
        return self.child.tick(context)

    def reset(self) -> None:
        self.child.reset()


class Inverter(Decorator):
    """反转节点 - 成功变失败，失败变成功"""

    def tick(self, context: Any) -> NodeStatus:
        status = self.child.tick(context)
        if status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING


class Repeater(Decorator):
    """重复节点 - 重复执行子节点"""

    def __init__(
        self,
        child: BehaviorNode,
        repeat_count: int = -1,
        name: str = "Repeater"
    ):
        super().__init__(child, name)
        self.repeat_count = repeat_count  # -1表示无限重复
        self.current_count = 0

    def tick(self, context: Any) -> NodeStatus:
        if self.repeat_count > 0 and self.current_count >= self.repeat_count:
            self.current_count = 0
            return NodeStatus.SUCCESS

        status = self.child.tick(context)
        if status != NodeStatus.RUNNING:
            self.current_count += 1
            if self.repeat_count == -1:
                return NodeStatus.RUNNING
            return self.tick(context)

        return NodeStatus.RUNNING

    def reset(self) -> None:
        self.current_count = 0
        self.child.reset()


class Succeeder(Decorator):
    """成功节点 - 无论子节点结果如何都返回成功"""

    def tick(self, context: Any) -> NodeStatus:
        self.child.tick(context)
        return NodeStatus.SUCCESS


class UntilFail(Decorator):
    """直到失败 - 重复执行直到子节点失败"""

    def tick(self, context: Any) -> NodeStatus:
        status = self.child.tick(context)
        if status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING


class UntilSuccess(Decorator):
    """直到成功 - 重复执行直到子节点成功"""

    def tick(self, context: Any) -> NodeStatus:
        status = self.child.tick(context)
        if status == NodeStatus.SUCCESS:
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING


# ========== 行为树构建器 ==========

class BehaviorTreeBuilder:
    """行为树构建器 - 流式API"""

    def __init__(self):
        self.root: Optional[BehaviorNode] = None
        self._stack: List[BehaviorNode] = []

    def sequence(self, name: str = "Sequence") -> 'BehaviorTreeBuilder':
        """添加顺序节点"""
        node = Sequence([], name)
        self._add_node(node)
        self._stack.append(node)
        return self

    def selector(self, name: str = "Selector") -> 'BehaviorTreeBuilder':
        """添加选择节点"""
        node = Selector([], name)
        self._add_node(node)
        self._stack.append(node)
        return self

    def action(self, func: Callable, name: str = "Action") -> 'BehaviorTreeBuilder':
        """添加动作节点"""
        node = Action(func, name)
        self._add_node(node)
        return self

    def condition(self, func: Callable, name: str = "Condition") -> 'BehaviorTreeBuilder':
        """添加条件节点"""
        node = Condition(func, name)
        self._add_node(node)
        return self

    def inverter(self, name: str = "Inverter") -> 'BehaviorTreeBuilder':
        """添加反转装饰器"""
        if not self._stack:
            raise ValueError("No node to invert")
        child = self._stack.pop()
        node = Inverter(child, name)
        self._add_node(node)
        return self

    def end(self) -> 'BehaviorTreeBuilder':
        """结束当前复合节点"""
        if self._stack:
            self._stack.pop()
        return self

    def build(self) -> BehaviorNode:
        """构建行为树"""
        if self.root is None:
            raise ValueError("No nodes added")
        return self.root

    def _add_node(self, node: BehaviorNode) -> None:
        if self.root is None:
            self.root = node
        elif self._stack:
            parent = self._stack[-1]
            if isinstance(parent, (Sequence, Selector)):
                parent.children.append(node)
                node.parent = parent


__all__ = [
    'NodeStatus',
    'BehaviorNode',
    'Sequence',
    'Selector',
    'Action',
    'Condition',
    'Decorator',
    'Inverter',
    'Repeater',
    'Succeeder',
    'UntilFail',
    'UntilSuccess',
    'BehaviorTreeBuilder'
]
