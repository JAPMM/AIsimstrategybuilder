"""
instant_optimizer.py
====================

This module implements a lightweight optimisation engine for golf shots.
Given a hole layout and a simple statistical model of the player's shot
patterns, the optimizer produces actionable recommendations for the
next shot.  It is intentionally pragmatic: rather than employing a
computationally expensive reinforcement learning model it uses a set
of heuristics derived from typical club distances and dispersion.

The optimizer exposes three key methods used by the backend API:

* :func:`get_optimal_shot` – return a single best recommendation for
  the next shot given the current position and lie.
* :func:`reoptimize_from_position` – recompute the optimal shot after
  the ball has been moved (e.g. when the user drags the ball on the
  front end).
* :func:`get_shot_options` – return a ranked list of alternative shot
  candidates for comparison.
"""

from __future__ import annotations

import math
import os
import csv
from typing import Dict, List, Tuple, Any


class InstantGolfOptimizer:
    """Optimise golf shots based on simple heuristics."""

    # Default club statistics (mean carry in yards, standard deviation of carry
    # in yards, standard deviation of lateral dispersion in yards).  These
    # values are approximations based on typical amateur distances and can be
    # overridden by supplying a custom csv file named
    # ``golf_shot_dispersion_summary.csv`` alongside this module.  The file
    # should contain columns ``Club``, ``mean_carry``, ``std_carry`` and
    # ``std_lateral``.
    DEFAULT_CLUB_STATS: Dict[str, Tuple[float, float, float]] = {
        "Driver": (250, 15, 20),
        "3 Wood": (230, 15, 18),
        "5 Wood": (215, 14, 17),
        "3 Hybrid": (205, 12, 15),
        "4 Iron": (195, 12, 13),
        "5 Iron": (185, 11, 12),
        "6 Iron": (175, 11, 11),
        "7 Iron": (165, 10, 10),
        "8 Iron": (155, 9, 9),
        "9 Iron": (145, 9, 8),
        "Pitching Wedge": (135, 8, 7),
        "Gap Wedge": (120, 8, 6),
        "Sand Wedge": (105, 7, 6),
        "Lob Wedge": (90, 7, 5),
    }

    def __init__(self, qnet: Any, hole_data: Dict[str, Any]):
        """Initialise with a (possibly dummy) Q‑network and hole layout."""
        self.qnet = qnet
        self.hole = hole_data
        self.club_stats = self._load_club_stats()
        # Extract pin position from hole_data; fall back to end of hole length
        pin_pos = hole_data.get("pin_position") or hole_data.get("pin_pos")
        if pin_pos and isinstance(pin_pos, (list, tuple)) and len(pin_pos) >= 2:
            self.pin_x, self.pin_y = pin_pos[0], pin_pos[1]
        else:
            # Use total hole length on x axis and 0 on y axis if unspecified
            length = hole_data.get("yardage") or hole_data.get("length") or 0
            self.pin_x, self.pin_y = length, 0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_club_stats(self) -> Dict[str, Tuple[float, float, float]]:
        """Load club dispersion statistics from a CSV if present.

        Returns a dictionary mapping club names to a tuple of
        (mean_carry, std_carry, std_lateral).  If the file cannot be
        read the default hard coded values are returned.
        """
        filename = os.path.join(os.path.dirname(__file__), "..", "golf_shot_dispersion_summary.csv")
        try:
            if os.path.exists(filename):
                stats: Dict[str, Tuple[float, float, float]] = {}
                with open(filename, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        club = row["Club"].strip()
                        mean_carry = float(row.get("mean_carry", 0))
                        std_carry = float(row.get("std_carry", 0))
                        std_lateral = float(row.get("std_lateral", 0))
                        stats[club] = (mean_carry, std_carry, std_lateral)
                # If file contained at least one row, return
                if stats:
                    return stats
        except Exception:
            # Ignore failures and fall back to defaults
            pass
        return self.DEFAULT_CLUB_STATS.copy()

    # ------------------------------------------------------------------
    # Optimisation logic
    # ------------------------------------------------------------------
    def get_optimal_shot(
        self,
        distance_to_pin: float,
        lateral_position: float = 0.0,
        shot_number: int = 1,
        lie_type: str = "Fairway",
        player_mode: str = "Normal",
    ) -> Dict[str, Any]:
        """Return the single best shot recommendation.

        A simple heuristic is used: we choose the club whose mean carry
        distance most closely matches (but does not exceed) the current
        distance to the pin, then adjust the aim to correct for lateral
        deviation.  If the distance is shorter than our shortest club
        distance we choose a wedge.  The returned dictionary includes
        the selected club, the aim angle, and expected dispersion values.
        """
        # Compute absolute horizontal distance remaining
        target_distance = max(distance_to_pin, 0.0)
        # Determine best club
        selected_club = self._choose_club(target_distance)
        mean_carry, std_carry, std_lateral = self.club_stats[selected_club]
        # Calculate recommended aim: attempt to compensate for current lateral position
        # by aiming opposite the miss.  The aim is expressed in degrees left(-)/right(+).
        aim_angle = 0.0
        if std_lateral > 0:
            aim_angle = -lateral_position / std_lateral * 5.0  # scale factor to convert yards to degrees
            # Clamp aim angle to realistic range [-20, 20] degrees
            aim_angle = max(min(aim_angle, 20.0), -20.0)
        result = {
            "club": selected_club,
            "aim_angle_degrees": round(aim_angle, 1),
            "expected_carry": mean_carry,
            "expected_lateral_std": std_lateral,
            "estimated_remaining_distance": max(target_distance - mean_carry, 0.0),
            "shot_number": shot_number,
            "lie_type": lie_type,
        }
        return result

    def _choose_club(self, distance: float) -> str:
        """Select the club whose mean carry is closest to the remaining distance.

        Preference is given to clubs that will not grossly over‑carry the
        target.  If all clubs overshoot, the one with the shortest mean
        carry is chosen.
        """
        best_club = None
        best_error = float("inf")
        for club, (mean_carry, _, _) in self.club_stats.items():
            error = abs(distance - mean_carry)
            # Penalise overshoot by adding a large constant to the error
            if mean_carry > distance:
                error += 50
            if error < best_error:
                best_error = error
                best_club = club
        return best_club or list(self.club_stats.keys())[0]

    def reoptimize_from_position(
        self, ball_x: float, ball_y: float, shot_number: int
    ) -> Dict[str, Any]:
        """Recompute the optimal shot based on the ball's current coordinates.

        The new distance to the pin is computed using Euclidean distance in
        the 2D plane defined by x (downrange) and y (lateral).  The method
        returns the same structure as :func:`get_optimal_shot`.
        """
        dx = self.pin_x - ball_x
        dy = self.pin_y - ball_y
        distance = math.sqrt(dx * dx + dy * dy)
        lateral = dy  # lateral position relative to target line
        return self.get_optimal_shot(distance, lateral, shot_number)

    def get_shot_options(
        self, distance_to_pin: float, lateral_position: float, shot_number: int, top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """Return a ranked list of alternative shot options.

        For each club we compute the absolute error between its mean carry
        distance and the target distance.  The top ``n`` clubs with the
        lowest error (overshoot penalty applied) are returned along with
        recommended aim angles.  This function is useful for scenarios
        where the user wants to compare options instead of taking the
        single top recommendation.
        """
        candidates: List[Tuple[str, float]] = []
        for club, (mean_carry, _, _) in self.club_stats.items():
            error = abs(distance_to_pin - mean_carry)
            if mean_carry > distance_to_pin:
                error += 50
            candidates.append((club, error))
        # Sort by error
        candidates.sort(key=lambda x: x[1])
        options = []
        for club, _ in candidates[: max(top_n, 1)]:
            mean_carry, _, std_lateral = self.club_stats[club]
            aim_angle = 0.0
            if std_lateral > 0:
                aim_angle = -lateral_position / std_lateral * 5.0
                aim_angle = max(min(aim_angle, 20.0), -20.0)
            options.append(
                {
                    "club": club,
                    "aim_angle_degrees": round(aim_angle, 1),
                    "expected_carry": mean_carry,
                    "expected_lateral_std": std_lateral,
                    "estimated_remaining_distance": max(distance_to_pin - mean_carry, 0.0),
                    "shot_number": shot_number,
                }
            )
        return options