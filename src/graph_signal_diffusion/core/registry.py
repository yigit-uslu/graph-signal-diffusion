# Registry pattern implementation for managing named objects.
from __future__ import annotations
from typing import Callable, Dict, TypeVar

T = TypeVar("T")

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if key in self._items:
                raise KeyError(f"[{self.name}] '{key}' already registered")
            self._items[key] = obj
            return obj
        return deco

    def get(self, key: str) -> T:
        if key not in self._items:
            known = ", ".join(sorted(self._items.keys()))
            raise KeyError(f"[{self.name}] unknown '{key}'. Known: {known}")
        return self._items[key]

    def keys(self):
        return sorted(self._items.keys())
