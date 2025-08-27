import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Import your existing components
from backend.your_previous_model import (
    club_means, club_stds, lateral_stds, clubs_all, 
    mode_encoding_map, classify_shot, Player
)

class InstantGolfOptimizer:
    def __init__(self, qnet, hole_data: Dict):
        self.qnet = qnet
        self.qnet.eval()  # Set to evaluation mode
        self.hole_data = hole_data
        self.clubs_all = clubs_all
        self.club_means = club_means
        self.player = Player(club_means, club_stds, lateral_stds)
        
        # Pre-compute available clubs for speed
        self.tee_clubs = [c for c in clubs_all if club_means[c] >= 180]
        self.approach_clubs = [c for c in clubs_all if c != "Driver"]
        
    def get_optimal_shot(self, 
                        distance_to_pin: float, 
                        lateral_position: float, 
                        shot_number: int = 1,
                        lie_type: str = "Fairway",
                        player_mode: str = "Normal") -> Dict:
        """
        Instantly return optimal shot given current position
        Uses trained Q-network for <10ms response time
        """
        
        # Get available clubs
        available_clubs = self.tee_clubs if shot_number == 1 else self.approach_clubs
        
        # Encode player mode
        mode_encoding = mode_encoding_map.get(player_mode, 2)
        
        best_qvalue = float('-inf')
        best_shot = None
        
        # Evaluate all possible club/aim combinations
        aim_options = [-15, -10, -5, 0, 5, 10, 15]
        
        with torch.no_grad():  # Disable gradients for inference speed
            for club in available_clubs:
                for aim in aim_options:
                    # Encode state-action pair
                    state_vector = self._encode_state_action(
                        0, distance_to_pin, lateral_position, 
                        shot_number, mode_encoding, club, aim
                    )
                    
                    # Get Q-value prediction
                    qvalue = self.qnet(torch.tensor(state_vector, dtype=torch.float32)).item()
                    
                    if qvalue > best_qvalue:
                        best_qvalue = qvalue
                        best_shot = {
                            "club": club,
                            "aim_point": aim,
                            "confidence": self._qvalue_to_confidence(qvalue),
                            "q_value": qvalue,
                            "expected_outcome": self._predict_shot_outcome(
                                club, aim, distance_to_pin, lateral_position, lie_type
                            ),
                            "strokes_gained": self._calculate_strokes_gained(qvalue, distance_to_pin)
                        }
        
        return best_shot
    
    def get_shot_options(self, 
                        distance_to_pin: float, 
                        lateral_position: float, 
                        shot_number: int = 1,
                        top_n: int = 3) -> List[Dict]:
        """Return top N shot options ranked by Q-value for strategy comparison"""
        
        available_clubs = self.tee_clubs if shot_number == 1 else self.approach_clubs
        mode_encoding = 2  # Normal mode
        
        options = []
        aim_options = [-15, -10, -5, 0, 5, 10, 15]
        
        with torch.no_grad():
            for club in available_clubs:
                for aim in aim_options:
                    state_vector = self._encode_state_action(
                        0, distance_to_pin, lateral_position, 
                        shot_number, mode_encoding, club, aim
                    )
                    
                    qvalue = self.qnet(torch.tensor(state_vector, dtype=torch.float32)).item()
                    
                    options.append({
                        "club": club,
                        "aim_point": aim,
                        "q_value": qvalue,
                        "confidence": self._qvalue_to_confidence(qvalue),
                        "expected_outcome": self._predict_shot_outcome(
                            club, aim, distance_to_pin, lateral_position, "Fairway"
                        ),
                        "strokes_gained": self._calculate_strokes_gained(qvalue, distance_to_pin),
                        "risk_level": self._assess_risk_level(club, aim, distance_to_pin)
                    })
        
        # Return top N options sorted by Q-value
        return sorted(options, key=lambda x: x['q_value'], reverse=True)[:top_n]
    
    def reoptimize_from_position(self, 
                               ball_x: float, 
                               ball_y: float, 
                               shot_number: int) -> Dict:
        """
        Instantly reoptimize strategy from new ball position
        Called when user drags ball to new location
        """
        
        # Calculate new distance to pin
        pin_x, pin_y = self.hole_data['pin_position']
        distance_to_pin = np.sqrt((ball_x - pin_x)**2 + (ball_y - pin_y)**2)
        lateral_position = ball_y  # Assuming fairway centerline is y=0
        
        # Determine lie type based on position
        lie_type = self._classify_lie_from_position(ball_x, ball_y)
        
        # Get optimal shot from new position
        optimal_shot = self.get_optimal_shot(
            distance_to_pin, lateral_position, shot_number, lie_type
        )
        
        # Get alternative options
        alternatives = self.get_shot_options(distance_to_pin, lateral_position, shot_number)
        
        return {
            "current_position": {"x": ball_x, "y": ball_y},
            "distance_to_pin": round(distance_to_pin, 1),
            "lateral_position": round(lateral_position, 1),
            "lie_type": lie_type,
            "optimal_shot": optimal_shot,
            "alternatives": alternatives[1:],  # Skip first (optimal) option
            "course_strategy": self._get_strategy_context(distance_to_pin, shot_number)
        }
    
    def _encode_state_action(self, hole_idx, distance, lateral, shot_num, mode_encoding, club, aim):
        """Same encoding as training - maintain consistency"""
        club_onehot = [1 if club == c else 0 for c in self.clubs_all]
        aim_norm = aim / 15
        return [hole_idx, distance/500, lateral/50, shot_num/10, mode_encoding/4] + club_onehot + [aim_norm]
    
    def _qvalue_to_confidence(self, qvalue: float) -> float:
        """Convert Q-value to confidence percentage (0-100)"""
        # Normalize based on typical Q-value ranges from training
        # Adjust these bounds based on your actual Q-value distribution
        min_q, max_q = -15.0, 5.0
        normalized = max(0, min(1, (qvalue - min_q) / (max_q - min_q)))
        return round(normalized * 100, 1)
    
    def _predict_shot_outcome(self, club: str, aim: int, distance: float, lateral: float, lie: str) -> Dict:
        """Predict expected outcome using same logic as training"""
        
        # Get base club performance
        expected_carry = self.club_means[club]
        carry_std = club_stds[club]
        lateral_std = lateral_stds[club]
        
        # Adjust for lie conditions (same as training)
        lie_adjustments = {
            "Tee": {"carry_mult": 1.0, "std_mult": 1.0},
            "Fairway": {"carry_mult": 1.0, "std_mult": 0.85},
            "First Cut": {"carry_mult": 0.95, "std_mult": 0.95},
            "Rough": {"carry_mult": 0.85, "std_mult": 1.05},
            "Deep Rough": {"carry_mult": 0.75, "std_mult": 1.18},
            "Bunker": {"carry_mult": 0.6, "std_mult": 1.2},
            "Tree": {"carry_mult": 0.5, "std_mult": 2.0},
            "Fringe": {"carry_mult": 0.95, "std_mult": 0.93}
        }
        
        adj = lie_adjustments.get(lie, {"carry_mult": 1.0, "std_mult": 1.0})
        adjusted_carry = expected_carry * adj["carry_mult"]
        adjusted_std = carry_std * adj["std_mult"]
        
        # Calculate expected final position
        remaining_distance = max(0, distance - adjusted_carry)
        expected_lateral = lateral + aim
        
        return {
            "carry_distance": round(adjusted_carry, 1),
            "carry_std": round(adjusted_std, 1),
            "remaining_distance": round(remaining_distance, 1),
            "expected_lateral": round(expected_lateral, 1),
            "success_probability": round(self._calculate_success_prob(club, distance, lie) * 100, 1),
            "expected_strokes_from_result": self._estimate_strokes_from_position(
                remaining_distance, abs(expected_lateral)
            )
        }
    
    def _calculate_success_prob(self, club: str, distance: float, lie: str) -> float:
        """Calculate probability of good outcome"""
        expected_carry = self.club_means[club]
        distance_diff = abs(expected_carry - distance)
        
        # Base probability based on club-distance match
        if distance_diff < 10:
            base_prob = 0.85
        elif distance_diff < 20:
            base_prob = 0.75
        elif distance_diff < 30:
            base_prob = 0.65
        else:
            base_prob = 0.5
        
        # Adjust for lie
        lie_multipliers = {
            "Tee": 1.0,
            "Fairway": 0.95,
            "First Cut": 0.85,
            "Rough": 0.75,
            "Deep Rough": 0.6,
            "Bunker": 0.5,
            "Tree": 0.4,
            "Fringe": 0.9
        }
        
        return base_prob * lie_multipliers.get(lie, 0.8)
    
    def _classify_lie_from_position(self, x: float, y: float) -> str:
        """Determine lie type based on ball position and hole geometry"""
        
        # Use same classification logic as training
        hole_dict = {
            'green_distance': self.hole_data.get('green_distance', 400),
            'green_depth': self.hole_data.get('green_depth', 25), 
            'green_width': self.hole_data.get('green_width', 20),
            'fairway_width': self.hole_data.get('fairway_width', 30),
            'zones': self.hole_data.get('zones', []),
            'ob_zones': self.hole_data.get('ob_zones', []),
            'water_zones': self.hole_data.get('water_zones', [])
        }
        
        # Create temporary hole object for classification
        class TempHole:
            def __init__(self, data):
                for key, value in data.items():
                    setattr(self, key, value)
        
        temp_hole = TempHole(hole_dict)
        return classify_shot(temp_hole, x, y)
    
    def _calculate_strokes_gained(self, qvalue: float, distance_to_pin: float) -> float:
        """Convert Q-value to strokes gained estimate"""
        # This is an approximation - you might want to calibrate this
        baseline_strokes = self._get_baseline_strokes(distance_to_pin)
        
        # Convert Q-value to strokes improvement
        # Higher Q-value = fewer strokes expected
        strokes_improvement = qvalue * 0.1  # Adjust multiplier based on your Q-value scale
        
        return round(strokes_improvement, 2)
    
    def _get_baseline_strokes(self, distance: float) -> float:
        """Baseline strokes expected from distance (PGA tour averages)"""
        if distance <= 3:
            return 1.0
        elif distance <= 10:
            return 1.1
        elif distance <= 25:
            return 1.3
        elif distance <= 50:
            return 1.8
        elif distance <= 100:
            return 2.4
        elif distance <= 150:
            return 2.8
        elif distance <= 200:
            return 3.2
        else:
            return 3.5 + (distance - 200) * 0.01
    
    def _assess_risk_level(self, club: str, aim: int, distance: float) -> str:
        """Assess risk level of shot choice"""
        expected_carry = self.club_means[club]
        distance_diff = abs(expected_carry - distance)
        
        # High risk if club doesn't match distance well
        if distance_diff > 30:
            return "High"
        elif distance_diff > 15:
            return "Medium" 
        elif abs(aim) > 10:
            return "Medium"  # Aggressive aim
        else:
            return "Low"
    
    def _estimate_strokes_from_position(self, distance: float, lateral_error: float) -> float:
        """Estimate strokes needed from resulting position"""
        base_strokes = self._get_baseline_strokes(distance)
        
        # Add penalty for lateral error
        if lateral_error > 30:
            base_strokes += 0.5
        elif lateral_error > 15:
            base_strokes += 0.2
        
        return round(base_strokes, 1)
    
    def _get_strategy_context(self, distance: float, shot_number: int) -> Dict:
        """Provide strategic context for the shot"""
        return {
            "shot_type": self._categorize_shot(distance, shot_number),
            "key_consideration": self._get_key_consideration(distance, shot_number),
            "target_zone": self._get_target_zone(distance)
        }
    
    def _categorize_shot(self, distance: float, shot_number: int) -> str:
        """Categorize the type of shot"""
        if shot_number == 1:
            return "Tee Shot"
        elif distance > 150:
            return "Approach Shot"
        elif distance > 50:
            return "Short Iron"
        elif distance > 20:
            return "Wedge"
        else:
            return "Short Game"
    
    def _get_key_consideration(self, distance: float, shot_number: int) -> str:
        """Get key strategic consideration"""
        if shot_number == 1:
            return "Position for approach shot"
        elif distance > 100:
            return "Hit the green"
        else:
            return "Get close to pin"
    
    def _get_target_zone(self, distance: float) -> str:
        """Get recommended target zone"""
        if distance > 150:
            return "Center of green"
        elif distance > 75:
            return "Pin-high, safe side"
        else:
            return "Attack the pin"