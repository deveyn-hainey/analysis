"""Stable ID assignment, per-team pruning, and possession smoothing."""
from typing import Any, Literal, Optional, Union

from . import config
from .geometry import pitch_distance
from .schemas import TeamId, TrackState


def prune_team_players(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one detection per track ID and cap each side to a plausible XI."""
    pruned: list[dict[str, Any]] = []

    for team in ("home", "away"):
        team_players = [p for p in players if p["team"] == team]
        best_by_id: dict[str, dict[str, Any]] = {}
        for player in team_players:
            existing = best_by_id.get(player["id"])
            player_score = float(player.get("detectionConfidence", 0.0)) * 1000 + float(player.get("boxArea", 0.0))
            existing_score = (
                float(existing.get("detectionConfidence", 0.0)) * 1000 + float(existing.get("boxArea", 0.0))
                if existing
                else -1
            )
            if existing is None or player_score > existing_score:
                best_by_id[player["id"]] = player

        kept = sorted(
            best_by_id.values(),
            key=lambda p: (float(p.get("detectionConfidence", 0.0)), float(p.get("boxArea", 0.0))),
            reverse=True,
        )[:config.MAX_PLAYERS_PER_TEAM]
        pruned.extend(sorted(kept, key=lambda p: p["id"]))

    return pruned


def stabilize_player_ids(
    frame: dict[str, Any],
    tracks: dict[TeamId, list[TrackState]],
    next_ids: dict[TeamId, int],
    max_jump: float = 20.0,
) -> dict[str, Any]:
    """Assign stable hN/aN IDs and smooth positions across tracker dropouts."""
    timestamp = float(frame["timestamp"])
    updated_players: list[dict[str, Any]] = []
    used_track_indices: dict[TeamId, set[int]] = {"home": set(), "away": set()}
    matched_existing = 0
    created_tracks = 0

    for team in ("home", "away"):
        team_players = [p for p in frame["players"] if p["team"] == team]

        for player in team_players:
            tracker_number = int(player["number"]) if player.get("number") else None
            canonical_team: TeamId = team
            if tracker_number is not None:
                for candidate_team in ("home", "away"):
                    if any(track.get("tracker_id") == tracker_number for track in tracks[candidate_team]):
                        canonical_team = candidate_team
                        break
                player["team"] = canonical_team

            team_tracks = tracks[canonical_team]

            best_idx: Optional[int] = None
            best_cost = max_jump
            for idx, track in enumerate(team_tracks):
                if idx in used_track_indices[canonical_team]:
                    continue
                if tracker_number is not None and track.get("tracker_id") == tracker_number:
                    best_idx = idx
                    best_cost = 0
                    break
                age = timestamp - float(track["last_seen"])
                if age > 10.0:
                    continue
                cost = pitch_distance(player["position"], track["position"]) + age * 1.5
                if cost < best_cost:
                    best_idx = idx
                    best_cost = cost

            if best_idx is None:
                if tracker_number is None:
                    next_ids[canonical_team] += 1
                    tracker_number = next_ids[canonical_team]
                else:
                    next_ids[canonical_team] = max(next_ids[canonical_team], tracker_number)
                stable_id = f"{'h' if canonical_team == 'home' else 'a'}{tracker_number}"
                team_tracks.append({
                    "id": stable_id,
                    "tracker_id": tracker_number,
                    "position": player["position"],
                    "smoothed_position": player["position"],
                    "last_seen": timestamp,
                })
                created_tracks += 1
            else:
                used_track_indices[canonical_team].add(best_idx)
                track = team_tracks[best_idx]
                stable_id = str(track["id"])
                previous_smoothed = track.get("smoothed_position", track["position"])
                alpha = config.TRACK_SMOOTHING_ALPHA
                smoothed = {
                    "x": round(previous_smoothed["x"] * (1 - alpha) + player["position"]["x"] * alpha, 1),
                    "y": round(previous_smoothed["y"] * (1 - alpha) + player["position"]["y"] * alpha, 1),
                }
                track["position"] = player["position"]
                track["smoothed_position"] = smoothed
                track["last_seen"] = timestamp
                matched_existing += 1

            player["id"] = stable_id
            try:
                player["number"] = int(stable_id[1:])
            except Exception:
                player["number"] = 0
            player["position"] = team_tracks[best_idx]["smoothed_position"] if best_idx is not None else player["position"]
            updated_players.append(player)

    raw_players = updated_players
    frame["players"] = prune_team_players(raw_players)
    if config.TRACK_DIAGNOSTICS:
        frame["_trackingDiagnostics"] = {
            "rawPlayers": len(raw_players),
            "postPrunePlayers": len(frame["players"]),
            "matchedExisting": matched_existing,
            "createdTracks": created_tracks,
            "droppedByPrune": max(0, len(raw_players) - len(frame["players"])),
            "homeCount": len([p for p in frame["players"] if p["team"] == "home"]),
            "awayCount": len([p for p in frame["players"] if p["team"] == "away"]),
            "stableIds": sorted(p["id"] for p in frame["players"]),
        }
    if frame.get("possessingPlayer"):
        poss = frame["possessingPlayer"]
        nearest = min(
            [p for p in frame["players"] if p["team"] == poss["team"]],
            key=lambda p: pitch_distance(p["position"], frame["ballPosition"]),
            default=None,
        )
        if nearest:
            frame["possessingPlayer"] = {"team": nearest["team"], "playerId": nearest["id"]}

    return frame


def smooth_possession(
    frame: dict[str, Any],
    previous_possession: Union[TeamId, Literal["contested"]],
    max_control_distance: float = 9.0,
) -> Union[TeamId, Literal["contested"]]:
    if not frame.get("ballPosition") or not frame["players"]:
        return previous_possession if previous_possession != "contested" else "contested"

    nearest = min(
        frame["players"],
        key=lambda p: pitch_distance(p["position"], frame["ballPosition"]),
    )
    distance = pitch_distance(nearest["position"], frame["ballPosition"])
    if distance > max_control_distance:
        return previous_possession if previous_possession != "contested" and distance <= max_control_distance * 1.6 else "contested"

    frame["possessingPlayer"] = {"team": nearest["team"], "playerId": nearest["id"]}
    return nearest["team"]
