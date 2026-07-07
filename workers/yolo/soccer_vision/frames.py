"""Per-frame analysis assembly: detections + teams → FrameData dict."""
from typing import Any, Literal, Optional, Union

import numpy as np
from PIL import Image

from .detection import decode_frame, detections_for_image, split_detections
from .geometry import box_area, player_role, position_from_box
from .pitch_homography import attach_pitch_positions
from .pitch_mask import compute_pitch_mask, filter_persons_to_pitch
from .schemas import Detection, RawFrame, TeamId
from .teams import assign_teams


def analyze_precomputed_frame(
    raw: RawFrame,
    frame_index: int,
    image: Image.Image,
    detections: list[Detection],
    teams: list[TeamId],
    is_pitch_view: bool = True,
    pitch_mask: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    width, height = image.size
    person_detections, ball_detections, referee_detections = split_detections(detections)

    players: list[dict[str, Any]] = []
    team_counts: dict[TeamId, int] = {"home": 0, "away": 0}

    for detection, team in zip(person_detections, teams):
        team_counts[team] += 1
        position = position_from_box(detection.xyxy, width, height, anchor="bottom")
        tracker_number = detection.tracker_id if detection.tracker_id is not None else team_counts[team]
        player_id = f"{'h' if team == 'home' else 'a'}{tracker_number}"
        players.append(
            {
                "id": player_id,
                "number": detection.tracker_id or 0,
                "team": team,
                "role": player_role(position, team),
                "position": position,
                "action": "standing",
                "detectionConfidence": round(detection.confidence, 3),
                "boxArea": round(box_area(detection.xyxy), 1),
            }
        )

    ball_position = None
    if ball_detections:
        best_ball = max(ball_detections, key=lambda d: d.confidence)
        ball_position = position_from_box(best_ball.xyxy, width, height)

    possession: Union[TeamId, Literal["contested"]] = "contested"
    possessing_player = None
    if ball_position and players:
        nearest = min(
            players,
            key=lambda p: (p["position"]["x"] - ball_position["x"]) ** 2
            + (p["position"]["y"] - ball_position["y"]) ** 2,
        )
        possession = nearest["team"]
        possessing_player = {"team": nearest["team"], "playerId": nearest["id"]}

    frame: dict[str, Any] = {
        "frameIndex": frame_index,
        "timestamp": raw.timestamp,
        "players": players,
        "ballPosition": ball_position,
        "isPitchView": is_pitch_view,
        "possession": possession,
        "events": [],
    }

    if possessing_player:
        frame["possessingPlayer"] = possessing_player

    if referee_detections:
        frame["referees"] = [position_from_box(d.xyxy, width, height) for d in referee_detections]

    attach_pitch_positions(frame, image)
    return frame


def analyze_single_frame(
    raw: RawFrame,
    frame_index: int,
    home_kit_color: Optional[str] = None,
    away_kit_color: Optional[str] = None,
) -> dict[str, Any]:
    """Decode, detect, team-assign, and assemble one standalone frame."""
    image = decode_frame(raw.base64)
    pitch_mask = compute_pitch_mask(image)
    detections = filter_persons_to_pitch(image, detections_for_image(image), pitch_mask)
    person_detections, _, _ = split_detections(detections)
    teams = assign_teams(image, person_detections, None, home_kit_color, away_kit_color)
    return analyze_precomputed_frame(
        raw, frame_index, image, detections, teams, pitch_mask is not None, pitch_mask
    )
