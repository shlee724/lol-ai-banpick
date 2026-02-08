# core/draft_schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


ROLE_ALIASES = {
    "top": "top",
    "jungle": "jungle",
    "jg": "jungle",
    "mid": "mid",
    "middle": "mid",
    "adc": "adc",
    "bot": "adc",
    "bottom": "adc",
    "support": "support",
    "sup": "support",
    "sp": "support",
}

CANON_ROLES = ["top", "jungle", "mid", "adc", "support"]


@dataclass
class PickedWithRoles:
    my_team: Dict[str, Optional[str]]      # role -> champion
    enemy_team: List[Optional[str]]        # len 5
    notes: str = ""


def normalize_picks_with_roles(data: Dict[str, Any]) -> PickedWithRoles:
    my = data.get("my_team") or {}
    enemy = data.get("enemy_team") or []
    notes = data.get("notes") or ""

    # my_team 정규화
    norm_my: Dict[str, Optional[str]] = {r: None for r in CANON_ROLES}

    if isinstance(my, dict):
        for k, v in my.items():
            if not isinstance(k, str):
                continue
            kk = ROLE_ALIASES.get(k.strip().lower())
            if kk in norm_my:
                norm_my[kk] = v if isinstance(v, str) and v.strip() else None

    # enemy_team 정규화 (5칸 맞추기)
    norm_enemy: List[Optional[str]] = []
    if isinstance(enemy, list):
        for item in enemy[:5]:
            norm_enemy.append(item if isinstance(item, str) and item.strip() else None)
    while len(norm_enemy) < 5:
        norm_enemy.append(None)

    return PickedWithRoles(my_team=norm_my, enemy_team=norm_enemy, notes=notes)
