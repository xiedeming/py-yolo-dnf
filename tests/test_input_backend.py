"""键盘输入后端回归测试。

曾出现的问题：InputController 全程用 pynput 发送**虚拟键码**，DirectInput 游戏
（DNF 等）不响应，表现为"按键不操作游戏"。pydirectinput 以**扫描码**发送，早已
写进 requirements 却从未被使用。
"""
import sys
import types
import unittest
from unittest import mock

from src.control.input_controller import (
    DirectInputKeyboard,
    InputController,
    PynputKeyboard,
    create_keyboard_backend,
    _DIRECTINPUT_KEYS,
    _PYNPUT_KEYS,
)


def _fake_pydirectinput(sent):
    fake = types.ModuleType('pydirectinput')
    fake.FAILSAFE = True
    fake.keyDown = lambda name: sent.append(('down', name))
    fake.keyUp = lambda name: sent.append(('up', name))
    return fake


class KeyNameTests(unittest.TestCase):
    def setUp(self):
        # 只建立后端，不发送任何输入
        self.controller = InputController()

    def test_every_special_key_resolves_in_both_backends(self):
        # 缺一个名字就会在游戏中途 KeyError
        for canonical in self.controller._special_keys.values():
            with self.subTest(key=canonical):
                self.assertIn(canonical, _DIRECTINPUT_KEYS)
                self.assertIn(canonical, _PYNPUT_KEYS)

    def test_esc_and_escape_normalize_to_the_same_key(self):
        self.assertEqual(self.controller._get_key('esc'), self.controller._get_key('escape'))

    def test_single_character_keys_pass_through(self):
        for char in 'qwxy':
            with self.subTest(char=char):
                self.assertEqual(self.controller._get_key(char), char)

    def test_unknown_key_name_raises(self):
        with self.assertRaises(ValueError):
            self.controller._get_key('not_a_key')


class BackendSelectionTests(unittest.TestCase):
    def test_prefers_pydirectinput_when_importable(self):
        with mock.patch.dict(sys.modules, {'pydirectinput': _fake_pydirectinput([])}):
            self.assertIsInstance(create_keyboard_backend(), DirectInputKeyboard)

    def test_falls_back_to_pynput_when_pydirectinput_missing(self):
        with mock.patch.dict(sys.modules, {'pydirectinput': None}):
            self.assertIsInstance(create_keyboard_backend(), PynputKeyboard)


class DirectInputTranslationTests(unittest.TestCase):
    def test_sends_scancode_names_not_virtual_keys(self):
        sent = []
        with mock.patch.dict(sys.modules, {'pydirectinput': _fake_pydirectinput(sent)}):
            backend = DirectInputKeyboard()
            backend.press('escape')
            backend.release('page_up')
            backend.press('x')

        self.assertEqual(sent, [('down', 'esc'), ('up', 'pageup'), ('down', 'x')])

    def test_disables_failsafe(self):
        # 默认鼠标移到屏幕角落会抛异常中断自动化
        fake = _fake_pydirectinput([])
        with mock.patch.dict(sys.modules, {'pydirectinput': fake}):
            DirectInputKeyboard()
        self.assertFalse(fake.FAILSAFE)


if __name__ == '__main__':
    unittest.main()
