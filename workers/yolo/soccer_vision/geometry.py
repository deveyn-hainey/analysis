"""Pure coordinate math shared across the pipeline."""
from typing import Literal

import numpy as np

from .schemas import TeamId


def position_from_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    anchor: Literal["center", "bottom"] = "center",
) -> dict[str, float]:
    x1, y1, x2, y2 = box
    x = (x1 + x2) / 2
    y = y2 if anchor == "bottom" else (y1 + y2) / 2
    return {
        "x": round((x / width) * 100, 1),
        "y": round((y / height) * 100, 1),
    }


def box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def pitch_distance(a: dict[str, float], b: dict[str, float]) -> float:
    return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))


def lerp_position(start: dict[str, float], end: dict[str, float], alpha: float) -> dict[str, float]:
    return {
        "x": round(start["x"] * (1 - alpha) + end["x"] * alpha, 1),
        "y": round(start["y"] * (1 - alpha) + end["y"] * alpha, 1),
    }


def mirror_pitch_position(position: dict[str, float]) -> dict[str, float]:
    return {"x": round(100.0 - position["x"], 1), "y": round(position["y"], 1)}


def pitch_position_distance(a: dict[str, float], b: dict[str, float], mirrored: bool = False) -> float:
    pos = mirror_pitch_position(a) if mirrored else a
    return pitch_distance(pos, b)


def player_role(position: dict[str, float], team: TeamId) -> str:
    x = position["x"]
    own_goal_x = 0 if team == "home" else 100
    distance_from_own_goal = abs(x - own_goal_x)
    if distance_from_own_goal < 12:
        return "gk"
    if distance_from_own_goal < 35:
        return "def"
    if distance_from_own_goal < 68:
        return "mid"
    return "fwd"
