"""Unit tests for the pure (no-model) parts of the soccer_vision pipeline.

Run from workers/yolo:  .venv/bin/python -m pytest tests -q
These import config/geometry/teams/tracking/postprocess only — no YOLO weights
are loaded, so they run in milliseconds.
"""
import numpy as np
import pytest

from soccer_vision.geometry import (
    box_area,
    lerp_position,
    mirror_pitch_position,
    pitch_distance,
    player_role,
    position_from_box,
)
from soccer_vision.postprocess import (
    consolidate_player_teams,
    interpolate_ball_positions,
    interpolate_player_positions,
)
from soccer_vision.teams import (
    assign_teams_by_known_kits,
    kit_color_signature,
    kmeans_two,
    normalize_kit_color,
    split_teams,
)
from soccer_vision.tracking import prune_team_players, smooth_possession


def make_player(pid, team, x=50.0, y=50.0, conf=0.9):
    return {
        "id": pid,
        "number": 0,
        "team": team,
        "role": "mid",
        "position": {"x": x, "y": y},
        "action": "standing",
        "detectionConfidence": conf,
        "boxArea": 100.0,
    }


def make_frame(idx, players, ball=None):
    return {
        "frameIndex": idx,
        "timestamp": float(idx),
        "players": players,
        "ballPosition": ball,
        "isPitchView": True,
        "possession": "contested",
        "events": [],
    }


class TestGeometry:
    def test_position_from_box_center_and_bottom(self):
        box = (0.0, 0.0, 100.0, 100.0)
        assert position_from_box(box, 200, 200) == {"x": 25.0, "y": 25.0}
        assert position_from_box(box, 200, 200, anchor="bottom") == {"x": 25.0, "y": 50.0}

    def test_box_area_clamps_negative(self):
        assert box_area((10, 10, 5, 20)) == 0.0
        assert box_area((0, 0, 4, 5)) == 20.0

    def test_lerp_and_mirror(self):
        assert lerp_position({"x": 0, "y": 0}, {"x": 10, "y": 20}, 0.5) == {"x": 5.0, "y": 10.0}
        assert mirror_pitch_position({"x": 30.0, "y": 40.0}) == {"x": 70.0, "y": 40.0}

    def test_player_role_bands(self):
        assert player_role({"x": 5, "y": 50}, "home") == "gk"
        assert player_role({"x": 20, "y": 50}, "home") == "def"
        assert player_role({"x": 50, "y": 50}, "home") == "mid"
        assert player_role({"x": 90, "y": 50}, "home") == "fwd"
        # Away attacks the other way
        assert player_role({"x": 95, "y": 50}, "away") == "gk"

    def test_pitch_distance(self):
        assert pitch_distance({"x": 0, "y": 0}, {"x": 3, "y": 4}) == 5.0


class TestTeams:
    def test_kmeans_two_separates_clusters(self):
        a = [np.array([0.0, 0.0]) + np.random.rand(2) * 0.1 for _ in range(5)]
        b = [np.array([10.0, 10.0]) + np.random.rand(2) * 0.1 for _ in range(5)]
        labels, centroids = kmeans_two(a + b)
        assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1
        assert labels[0] != labels[5]

    def test_split_teams_edge_cases(self):
        assert split_teams([]) == []
        assert split_teams([np.zeros(4)]) == ["home"]

    def test_normalize_kit_color(self):
        assert normalize_kit_color(" Red ") == "red"
        assert normalize_kit_color("auto") == ""
        assert normalize_kit_color(None) == ""

    def test_known_kit_anchoring_labels_by_color(self):
        red = kit_color_signature("red")
        blue = kit_color_signature("blue")
        sigs = [red.copy(), blue.copy(), red.copy()]
        teams = assign_teams_by_known_kits(sigs, "red", "blue")
        assert teams == ["home", "away", "home"]

    def test_known_kit_anchoring_requires_both_colors(self):
        assert assign_teams_by_known_kits([np.zeros(4)], "red", "") is None


class TestTracking:
    def test_prune_caps_each_team_to_xi(self):
        players = [make_player(f"h{i}", "home") for i in range(15)]
        players += [make_player(f"a{i}", "away") for i in range(3)]
        pruned = prune_team_players(players)
        assert len([p for p in pruned if p["team"] == "home"]) == 11
        assert len([p for p in pruned if p["team"] == "away"]) == 3

    def test_prune_keeps_best_duplicate_per_id(self):
        weak = make_player("h1", "home", conf=0.3)
        strong = make_player("h1", "home", conf=0.9)
        pruned = prune_team_players([weak, strong])
        assert len(pruned) == 1 and pruned[0]["detectionConfidence"] == 0.9

    def test_smooth_possession_assigns_nearest(self):
        frame = make_frame(0, [make_player("h1", "home", x=50, y=50)], ball={"x": 51, "y": 50})
        assert smooth_possession(frame, "contested") == "home"
        assert frame["possessingPlayer"]["playerId"] == "h1"

    def test_smooth_possession_holds_previous_when_ball_far(self):
        frame = make_frame(0, [make_player("h1", "home", x=0, y=0)], ball={"x": 90, "y": 90})
        assert smooth_possession(frame, "contested") == "contested"


class TestPostprocess:
    def test_ball_interpolation_fills_gap(self):
        frames = [
            make_frame(0, [], ball={"x": 0.0, "y": 0.0}),
            make_frame(1, [], ball=None),
            make_frame(2, [], ball={"x": 10.0, "y": 10.0}),
        ]
        out = interpolate_ball_positions(frames)
        assert out[1]["ballPosition"] == {"x": 5.0, "y": 5.0}
        assert out[1]["ballInterpolated"] is True

    def test_ball_interpolation_respects_limit(self):
        frames = [make_frame(0, [], ball={"x": 0.0, "y": 0.0})]
        frames += [make_frame(i, [], ball=None) for i in range(1, 5)]
        frames += [make_frame(5, [], ball={"x": 10.0, "y": 10.0})]
        out = interpolate_ball_positions(frames, limit=2)
        assert all(f["ballPosition"] is None for f in out[1:5])

    def test_player_interpolation_bridges_dropout(self):
        p0 = make_player("h1", "home", x=0.0, y=0.0)
        p2 = make_player("h1", "home", x=10.0, y=10.0)
        frames = [make_frame(0, [p0]), make_frame(1, []), make_frame(2, [p2])]
        out = interpolate_player_positions(frames)
        mid = out[1]["players"]
        assert len(mid) == 1 and mid[0]["inferred"] is True
        assert mid[0]["position"] == {"x": 5.0, "y": 5.0}

    def test_consolidate_locks_majority_team(self):
        frames = [
            make_frame(0, [make_player("h1", "home", conf=0.9)]),
            make_frame(1, [make_player("h1", "away", conf=0.2)]),
            make_frame(2, [make_player("h1", "home", conf=0.9)]),
        ]
        out = consolidate_player_teams(frames)
        assert all(f["players"][0]["team"] == "home" for f in out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


class TestBallTracker:
    def make_ball(self, x, y, conf, size=10.0):
        from soccer_vision.schemas import Detection
        return Detection("ball", conf, (x - size / 2, y - size / 2, x + size / 2, y + size / 2))

    def test_select_none_when_empty(self):
        from soccer_vision.ball import BallTracker
        assert BallTracker().select([], 1920, 1080) is None

    def test_select_prefers_confidence_without_history(self):
        from soccer_vision.ball import BallTracker
        t = BallTracker()
        weak, strong = self.make_ball(100, 100, 0.2), self.make_ball(900, 500, 0.6)
        assert t.select([weak, strong], 1920, 1080) is strong
        assert t.last_center == (900.0, 500.0)

    def test_select_prefers_nearby_over_distant_similar_confidence(self):
        from soccer_vision.ball import BallTracker
        t = BallTracker()
        t.select([self.make_ball(900, 500, 0.6)], 1920, 1080)
        near, far = self.make_ball(920, 510, 0.30), self.make_ball(100, 100, 0.35)
        assert t.select([near, far], 1920, 1080) is near

    def test_low_confidence_teleport_rejected(self):
        from soccer_vision.ball import BallTracker
        t = BallTracker()
        t.select([self.make_ball(900, 500, 0.6)], 1920, 1080)
        teleport = self.make_ball(50, 50, 0.11)
        assert t.select([teleport], 1920, 1080) is None

    def test_miss_counter_resets_on_accept(self):
        from soccer_vision.ball import BallTracker
        t = BallTracker()
        t.misses = 5
        t.select([self.make_ball(900, 500, 0.6)], 1920, 1080)
        assert t.misses == 0
