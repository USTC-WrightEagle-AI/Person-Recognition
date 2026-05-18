"""
Analyzer - 命令解析 + 按需调度子模块

接收 brain 发来的指令 JSON，解析出需要检索的人物属性，
按代价排序后逐层过滤检测到的人。

代价顺序: posture (~5ms) → gesture (~5ms) → clothing (后续实现 ~100ms)
"""

from typing import Optional


class Analyzer:
    """属性过滤管线，按代价排序串联过滤"""

    def __init__(self):
        self._cost_order = ["posture", "gesture", "cloth_color", "identity"]

    def filter_by_attributes(
        self, people: list, attributes: Optional[dict]
    ) -> list:
        """
        Args:
            people: 当前帧检测到的所有人（含 bbox, landmarks, posture, gesture 等）
            attributes: 要找的特征，如 {"posture": "sitting", "gesture": "waving"}
                       为 None 或空则原样返回

        Returns:
            过滤后的人列表（符合全部特征）
        """
        if not attributes:
            return people

        matched = list(people)
        for attr_name in self._cost_order:
            if attr_name not in attributes or not matched:
                continue
            expected = attributes[attr_name]
            if expected is None or expected == "":
                continue
            expected_lower = str(expected).lower()
            matched = [
                p for p in matched
                if str(p.get(attr_name, "")).lower() == expected_lower
            ]

        return matched
