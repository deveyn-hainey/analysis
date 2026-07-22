"""Clip-level cleanup passes run after all frames are analyzed.

Order matters (see run_postprocessing): team consolidation first so
interpolation copies the corrected team, orientation fixes before pitch-gap
filling, possession recomputed last over the fully-interpolated data.
"""
from typing import Any, Literal, Union

from . import config
from .geometry import lerp_position, mirror_pitch_position, pitch_position_distance, player_role
from .schemas import TeamId
from .tracking import prune_team_players, smooth_possession


def interpolate_ball_positions(
    frames: list[dict[str, Any]], limit: int = config.BALL_INTERPOLATION_LIMIT
) -> list[dict[str, Any]]:
    """Fill short gaps in ball detections, matching the cleaner offline demo behavior."""
    known = [(i, frame.get("ballPosition")) for i, frame in enumerate(frames) if frame.get("ballPosition")]
    if len(known) < 2:
        return frames

    for (start_idx, start_pos), (end_idx, end_pos) in zip(known, known[1:]):
        if start_pos is None or end_pos is None:
            continue
        gap = end_idx - start_idx - 1
        if gap <= 0 or gap > limit:
            continue
        for offset in range(1, gap + 1):
            alpha = offset / (gap + 1)
            frames[start_idx + offset]["ballPosition"] = lerp_position(start_pos, end_pos, alpha)
            start_pitch = frames[start_idx].get("pitchBall")
            end_pitch = frames[end_idx].get("pitchBall")
            if start_pitch and end_pitch:
                frames[start_idx + offset]["pitchBall"] = lerp_position(start_pitch, end_pitch, alpha)
            frames[start_idx + offset]["ballInterpolated"] = True

    return frames


def mirror_frame_pitch_positions(frame: dict[str, Any]) -> None:
    for player in frame.get("players", []):
        if player.get("pitchPosition"):
            player["pitchPosition"] = mirror_pitch_position(player["pitchPosition"])
    if frame.get("pitchBall"):
        frame["pitchBall"] = mirror_pitch_position(frame["pitchBall"])
    if frame.get("pitchReferees"):
        frame["pitchReferees"] = [mirror_pitch_position(pos) for pos in frame["pitchReferees"]]
    frame["pitchMirroredCorrection"] = True


def stabilize_pitch_orientation(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Correct occasional left/right homography flips after camera-angle changes."""
    previous_by_id: dict[str, dict[str, float]] = {}

    for frame in frames:
        current = {
            player["id"]: player["pitchPosition"]
            for player in frame.get("players", [])
            if player.get("pitchPosition")
        }
        shared = [(pid, pos, previous_by_id[pid]) for pid, pos in current.items() if pid in previous_by_id]

        if len(shared) >= 3:
            normal = sum(pitch_position_distance(pos, prev) for _, pos, prev in shared) / len(shared)
            mirrored = sum(pitch_position_distance(pos, prev, mirrored=True) for _, pos, prev in shared) / len(shared)
            if normal > 25 and mirrored + 8 < normal:
                mirror_frame_pitch_positions(frame)
                current = {
                    player["id"]: player["pitchPosition"]
                    for player in frame.get("players", [])
                    if player.get("pitchPosition")
                }

        if current:
            previous_by_id.update(current)

    return frames


def consolidate_player_teams(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Lock each stable player ID to a single team for the whole clip.

    Per-frame jersey clustering can occasionally flip a player home<->away for a
    single frame, which makes them jump across the tactical view. We tally a
    confidence-weighted vote per ID across every frame it appears in and rewrite
    the team (and role) to the majority, so a given player keeps one colour for
    the entire clip.
    """
    votes: dict[str, dict[TeamId, float]] = {}
    for frame in frames:
        for player in frame["players"]:
            weight = float(player.get("detectionConfidence", 0.0)) or 1.0
            tally = votes.setdefault(player["id"], {"home": 0.0, "away": 0.0})
            tally[player["team"]] += weight

    majority: dict[str, TeamId] = {
        pid: ("home" if tally["home"] >= tally["away"] else "away")
        for pid, tally in votes.items()
    }

    for frame in frames:
        for player in frame["players"]:
            team = majority.get(player["id"], player["team"])
            player["team"] = team
            player["role"] = player_role(player["position"], team)
        if frame.get("possessingPlayer"):
            pid = frame["possessingPlayer"].get("playerId")
            if pid in majority:
                frame["possessingPlayer"]["team"] = majority[pid]
    return frames


def interpolate_player_positions(
    frames: list[dict[str, Any]], limit: int = config.PLAYER_INTERPOLATION_LIMIT
) -> list[dict[str, Any]]:
    """Bridge short detection gaps per player so tracks don't flicker on/off.

    YOLO misses the odd frame when players overlap or blur; ByteTrack keeps the ID
    alive internally but emits no box, so the player vanishes for a frame or two and
    the overlay flashes. For each stable ID we linearly interpolate a position into
    the missing frames between two real sightings (gap <= limit), mark it inferred,
    and re-cap each frame to a plausible XI.
    """
    appearances: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, frame in enumerate(frames):
        for player in frame["players"]:
            appearances.setdefault(player["id"], []).append((idx, player))

    for pid, seen in appearances.items():
        for (start_idx, start_p), (end_idx, end_p) in zip(seen, seen[1:]):
            gap = end_idx - start_idx - 1
            if gap <= 0 or gap > limit:
                continue
            if any(frames[start_idx + offset].get("isPitchView") is False for offset in range(1, gap + 1)):
                continue
            for offset in range(1, gap + 1):
                alpha = offset / (gap + 1)
                position = lerp_position(start_p["position"], end_p["position"], alpha)
                player = {
                    **start_p,
                    "position": position,
                    "role": player_role(position, start_p["team"]),
                    # Slightly below a real detection so pruning prefers live boxes.
                    "detectionConfidence": round(float(start_p.get("detectionConfidence", 0.5)) * 0.9, 3),
                    "inferred": True,
                }
                if start_p.get("pitchPosition") and end_p.get("pitchPosition"):
                    player["pitchPosition"] = lerp_position(
                        start_p["pitchPosition"], end_p["pitchPosition"], alpha
                    )
                frames[start_idx + offset]["players"].append(player)

    for frame in frames:
        frame["players"] = prune_team_players(frame["players"])
    return frames


def fill_missing_pitch_positions(
    frames: list[dict[str, Any]],
    limit: int = max(config.PLAYER_INTERPOLATION_LIMIT, config.BALL_INTERPOLATION_LIMIT),
) -> list[dict[str, Any]]:
    """Fill short calibrated-coordinate gaps for existing tracks after cutaways."""
    appearances: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, frame in enumerate(frames):
        for player in frame.get("players", []):
            if player.get("pitchPosition"):
                appearances.setdefault(player["id"], []).append((idx, player))

    for pid, seen in appearances.items():
        for (start_idx, start_p), (end_idx, end_p) in zip(seen, seen[1:]):
            gap = end_idx - start_idx - 1
            if gap <= 0 or gap > limit:
                continue
            if any(frames[start_idx + offset].get("isPitchView") is False for offset in range(1, gap + 1)):
                continue
            for offset in range(1, gap + 1):
                player = next(
                    (p for p in frames[start_idx + offset].get("players", []) if p.get("id") == pid),
                    None,
                )
                if not player or player.get("pitchPosition"):
                    continue
                alpha = offset / (gap + 1)
                player["pitchPosition"] = lerp_position(
                    start_p["pitchPosition"], end_p["pitchPosition"], alpha
                )
                player["pitchInterpolated"] = True

    known_ball = [(i, frame.get("pitchBall")) for i, frame in enumerate(frames) if frame.get("pitchBall")]
    for (start_idx, start_pos), (end_idx, end_pos) in zip(known_ball, known_ball[1:]):
        if start_pos is None or end_pos is None:
            continue
        gap = end_idx - start_idx - 1
        if gap <= 0 or gap > limit:
            continue
        if any(frames[start_idx + offset].get("isPitchView") is False for offset in range(1, gap + 1)):
            continue
        for offset in range(1, gap + 1):
            frame = frames[start_idx + offset]
            if frame.get("pitchBall") or not frame.get("ballPosition"):
                continue
            alpha = offset / (gap + 1)
            frame["pitchBall"] = lerp_position(start_pos, end_pos, alpha)
            frame["pitchInterpolated"] = True

    return frames


def recompute_possession_for_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_possession: Union[TeamId, Literal["contested"]] = "contested"
    for frame in frames:
        frame["possession"] = smooth_possession(frame, previous_possession)
        previous_possession = frame["possession"]
    return frames


def run_postprocessing(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The standard clip-level cleanup sequence used by every endpoint."""
    frames = consolidate_player_teams(frames)
    frames = stabilize_pitch_orientation(frames)
    frames = interpolate_player_positions(frames)
    frames = fill_missing_pitch_positions(frames)
    return recompute_possession_for_frames(interpolate_ball_positions(frames))
