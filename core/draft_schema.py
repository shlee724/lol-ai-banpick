# core/draft_schema.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
    my_team: Dict[str, Optional[str]]  # role -> champion
    enemy_team: List[Optional[str]]  # len 5
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


@dataclass
class Bans10:
    bans: List[Optional[str]]  # len 10
    notes: str = ""


def normalize_bans10(data: Dict[str, Any]) -> Bans10:
    bans = data.get("bans") or []
    notes = data.get("notes") or ""

    norm: List[Optional[str]] = []
    if isinstance(bans, list):
        for x in bans[:10]:
            norm.append(x if isinstance(x, str) and x.strip() else None)

    while len(norm) < 10:
        norm.append(None)

    return Bans10(bans=norm, notes=notes)


def safe_get_draft_fields(res):
    """
    main.py와 동일한 기대 스키마:
      {
        "my_team": {...},
        "enemy_team": {...},
        "reco": {...}
      }
    """
    if isinstance(res, str):
        try:
            res = json.loads(res)
        except Exception:
            return None, None, None, {"_error": "json_parse_failed", "_raw": res}

    if not isinstance(res, dict):
        return None, None, None, {"_error": "not_a_dict", "_raw": res}

    my_team = res.get("my_team")
    enemy_team = res.get("enemy_team")
    reco = res.get("reco")
    if my_team is None or enemy_team is None or reco is None:
        return None, None, None, {"_error": "missing_keys", "_keys": list(res.keys()), "_raw": res}

    return my_team, enemy_team, reco, None
