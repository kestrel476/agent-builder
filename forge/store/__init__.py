"""Хранение: каждый агент в процессе создания — это рабочая папка на диске."""

from __future__ import annotations

from .workspace import Workspace, WorkspaceManager, slugify

__all__ = ["Workspace", "WorkspaceManager", "slugify"]
