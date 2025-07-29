import csv
import uuid
from datetime import datetime

def parse_trackman_csv(file_path):
    shots = []

    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            shot = {
                "shot_id": str(uuid.uuid4()),
                "player_id": None,
                "session_id": None,
                "timestamp": datetime.utcnow().isoformat(),

                # Club metadata
                "club": row.get("Club", "").strip(),
                "swing_type": row.get("Swing Type", "Full").strip(),
                "hole_number": int(row.get("Hole", 0) or 0),
                "shot_number": int(row.get("Shot", 0) or 0),

                # Ball Flight
                "ball_speed": try_float(row.get("Ball Speed")),
                "launch_angle": try_float(row.get("Launch Angle")),
                "launch_direction": try_float(row.get("Launch Direction")),
                "apex_height": try_float(row.get("Apex Height")),
                "spin_axis": try_float(row.get("Spin Axis")),
                "total_spin": try_float(row.get("Spin Rate")),

                # Distance
                "carry_distance": try_float(row.get("Carry")),
                "total_distance": try_float(row.get("Total")),
                "side": try_float(row.get("Side")),
                "side_total": try_float(row.get("Side Total")),
                "hang_time": try_float(row.get("Hang Time")),
                "landing_angle": try_float(row.get("Landing Angle")),

                # Club & Face
                "club_speed": try_float(row.get("Club Speed")),
                "attack_angle": try_float(row.get("Attack Angle")),
                "dynamic_loft": try_float(row.get("Dynamic Loft")),
                "face_angle": try_float(row.get("Face Angle")),
                "club_path": try_float(row.get("Club Path")),

                # Derived
                "smash_factor": try_float(row.get("Smash Factor")),
                "spin_loft": try_float(row.get("Spin Loft")),
                "speed_drop": try_float(row.get("Speed Drop")),

                # Swing Kinematics
                "swing_plane": try_float(row.get("Swing Plane")),
                "swing_direction": try_float(row.get("Swing Direction")),
                "swing_radius": try_float(row.get("Swing Radius")),
                "backswing_time": try_float(row.get("Backswing Time")),
                "downswing_time": try_float(row.get("Downswing Time")),

                # Advanced
                "low_point_height": try_float(row.get("Low Point Height")),
                "low_point_side": try_float(row.get("Low Point Side")),
                "d_plane_tilt": try_float(row.get("D Plane Tilt")),

                # Spatial (optional – can be inferred later)
                "landing_x": None,
                "landing_y": None,
                "resting_x": None,
                "resting_y": None,

                # Conditions
                "lie_type": row.get("Lie", "Tee"),
                "wind_speed": try_float(row.get("Wind Speed")),
                "wind_direction": try_float(row.get("Wind Direction")),
                "temperature": try_float(row.get("Temperature")),
                "humidity": try_float(row.get("Humidity")),
                "altitude": try_float(row.get("Altitude"))
            }

            shots.append(shot)

    return shots


def try_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
