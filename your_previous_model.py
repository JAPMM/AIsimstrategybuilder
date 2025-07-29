import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import Counter, deque, namedtuple
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

# ====== CONSTANTS ======
COURSE_GRID_SIZE = 64
MAX_SHOTS_PER_HOLE = 15
ELITE_LEARNING_RATE = 0.0003
ELITE_BATCH_SIZE = 128
MEMORY_CAPACITY = 500000

# Golf-specific constants
WIND_FACTOR_MAX = 0.3
PRESSURE_SITUATIONS = ["normal", "clutch", "pressure", "momentum_positive", "momentum_negative"]
SHOT_SHAPES = ["straight", "draw", "fade"]
TRAJECTORIES = ["low", "medium", "high"]
SWING_TEMPOS = ["smooth", "normal", "aggressive"]

# ====== ENHANCED DATA STRUCTURES ======
@dataclass
class WindCondition:
    speed: float  # mph
    direction: float  # degrees (0 = helping, 180 = into)
    gusts: float  # variability

@dataclass
class ShotResult:
    carry: float
    lateral: float
    final_position: Tuple[float, float]
    classification: str
    strike_quality: str
    ball_flight: str

@dataclass
class GameContext:
    round_number: int
    hole_number: int
    current_score_vs_par: int
    tournament_position: str
    pressure_level: float
    confidence: float
    momentum: float

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'priority'])

# ====== CLUB DATA LOAD (Enhanced) ======
def load_club_data():
    try:
        summary_df = pd.read_csv("golf_shot_dispersion_summary.csv")
    except FileNotFoundError:
        # Fallback data if CSV not found
        clubs = ["Driver", "3 Wood", "5 Wood", "3 Iron", "4 Iron", "5 Iron", 
                "6 Iron", "7 Iron", "8 Iron", "9 Iron", "Pitching Wedge", 
                "50* Wedge", "54* Wedge", "62* Wedge"]
        carries = [280, 245, 230, 210, 200, 190, 180, 170, 160, 150, 140, 120, 100, 80]
        stds = [25, 20, 18, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
        lat_stds = [35, 25, 22, 18, 16, 14, 12, 10, 8, 7, 6, 5, 4, 3]
        
        summary_df = pd.DataFrame({
            'Club': clubs,
            'mean_carry': carries,
            'std_carry': stds,
            'std_lateral': lat_stds
        })
    
    clubs_all = summary_df["Club"].tolist()
    club_means = dict(zip(summary_df["Club"], summary_df["mean_carry"]))
    club_stds = dict(zip(summary_df["Club"], summary_df["std_carry"]))
    lateral_stds = dict(zip(summary_df["Club"], summary_df["std_lateral"]))
    
    tee_clubs = [c for c in clubs_all if club_means[c] >= 180]
    wedge_names = ["Pitching Wedge", "50* Wedge", "54* Wedge", "62* Wedge"]
    
    return clubs_all, club_means, club_stds, lateral_stds, tee_clubs, wedge_names

clubs_all, club_means, club_stds, lateral_stds, tee_clubs, wedge_names = load_club_data()

# Enhanced mode modifiers with psychological factors
mode_modifiers = {
    "VeryGood": {"carry_adj": +15, "std_mult": 0.75, "confidence_boost": 0.2},
    "Good": {"carry_adj": +7, "std_mult": 0.85, "confidence_boost": 0.1},
    "Normal": {"carry_adj": 0, "std_mult": 1.0, "confidence_boost": 0.0},
    "Bad": {"carry_adj": -12, "std_mult": 1.4, "confidence_boost": -0.15},
    "Pressure": {"carry_adj": -5, "std_mult": 1.2, "confidence_boost": -0.1}
}

# ====== ADVANCED NEURAL ARCHITECTURE ======
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class CourseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Process 2D course representation
        self.conv1 = nn.Conv2d(6, 32, kernel_size=5, padding=2)  # 6 channels: fairway, rough, hazards, green, elevation, wind
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.attention1 = SpatialAttention(64)
        self.attention2 = SpatialAttention(128)
        
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(128 * 8 * 8, 512)
        
    def forward(self, course_map):
        x = F.relu(self.conv1(course_map))
        x = F.relu(self.conv2(x))
        x = self.attention1(x)
        x = F.relu(self.conv3(x))
        x = self.attention2(x)
        
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

class StrategyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.hidden_size = hidden_size
        
    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        return output[:, -1, :], hidden  # Return last output and hidden state

class DuelingQNetwork(nn.Module):
    def __init__(self, course_features=512, shot_history_features=256, context_features=64):
        super().__init__()
        
        # Course understanding
        self.course_encoder = CourseEncoder()
        
        # Shot sequence memory
        self.shot_lstm = StrategyLSTM(32, shot_history_features)
        
        # Context processing (wind, pressure, etc.)
        self.context_fc = nn.Sequential(
            nn.Linear(20, 128),
            nn.ReLU(),
            nn.Linear(128, context_features)
        )
        
        # Combined feature processing
        total_features = course_features + shot_history_features + context_features
        
        # Dueling architecture
        self.value_stream = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(clubs_all) * 21 * 3 * 3 * 3)  # clubs * aims * shapes * trajectories * tempos
        )
        
    def forward(self, course_map, shot_history, context, lstm_hidden=None):
        # Process different input streams
        course_features = self.course_encoder(course_map)
        shot_features, new_hidden = self.shot_lstm(shot_history, lstm_hidden)
        context_features = self.context_fc(context)
        
        # Combine all features
        combined = torch.cat([course_features, shot_features, context_features], dim=1)
        
        # Dueling streams
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        
        # Combine dueling streams
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - advantage_mean)
        
        return q_values, new_hidden

# ====== ENHANCED ENVIRONMENT ======
class EliteGolfHole:
    def __init__(self, length, fairway_width, green_distance, green_depth, green_width, 
                 elevation, zones, par, pin_pos, ob_zones=None, water_zones=None,
                 wind_condition=None, course_rating=72.0, slope_rating=113):
        self.length = length
        self.fairway_width = fairway_width
        self.green_distance = green_distance
        self.green_depth = green_depth
        self.green_width = green_width
        self.elevation = elevation
        self.zones = zones or []
        self.par = par
        self.pin_pos = pin_pos
        self.ob_zones = ob_zones or []
        self.water_zones = water_zones or []
        self.wind_condition = wind_condition or WindCondition(0, 0, 0)
        self.course_rating = course_rating
        self.slope_rating = slope_rating
        
    def create_2d_representation(self):
        """Create a 2D spatial representation of the hole"""
        course_map = np.zeros((6, COURSE_GRID_SIZE, COURSE_GRID_SIZE))
        
        # Map coordinates to grid
        x_scale = self.green_distance / COURSE_GRID_SIZE
        y_scale = 100 / COURSE_GRID_SIZE  # Assume max width of 100 yards
        
        # Channel 0: Fairway
        fairway_start_y = int((50 - self.fairway_width/2) / y_scale)
        fairway_end_y = int((50 + self.fairway_width/2) / y_scale)
        course_map[0, :, fairway_start_y:fairway_end_y] = 1.0
        
        # Channel 1: Rough (surrounding fairway)
        rough_start_y = max(0, int((50 - self.fairway_width/2 - 15) / y_scale))
        rough_end_y = min(COURSE_GRID_SIZE, int((50 + self.fairway_width/2 + 15) / y_scale))
        course_map[1, :, rough_start_y:rough_end_y] = 1.0
        course_map[1, :, fairway_start_y:fairway_end_y] = 0.0  # Remove fairway from rough
        
        # Channel 2: Hazards
        for zone in self.zones + self.water_zones + self.ob_zones:
            x1 = int(zone.get("x_start", 0) / x_scale)
            x2 = int(zone.get("x_end", 0) / x_scale)
            y1 = int((zone.get("y_start", 0) + 50) / y_scale)
            y2 = int((zone.get("y_end", 0) + 50) / y_scale)
            
            x1, x2 = max(0, min(x1, x2)), min(COURSE_GRID_SIZE, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(COURSE_GRID_SIZE, max(y1, y2))
            
            if zone.get("type") in ["Water", "OB"]:
                course_map[2, x1:x2, y1:y2] = 1.0  # High penalty
            else:
                course_map[2, x1:x2, y1:y2] = 0.5  # Moderate penalty
        
        # Channel 3: Green
        green_x1 = int((self.green_distance - self.green_depth/2) / x_scale)
        green_x2 = int((self.green_distance + self.green_depth/2) / x_scale)
        green_y1 = int((50 - self.green_width/2) / y_scale)
        green_y2 = int((50 + self.green_width/2) / y_scale)
        course_map[3, green_x1:green_x2, green_y1:green_y2] = 1.0
        
        # Channel 4: Elevation (simplified)
        elevation_factor = self.elevation / 30.0  # Normalize to [-1, 1]
        course_map[4, :, :] = elevation_factor
        
        # Channel 5: Wind effect zones
        if self.wind_condition.speed > 5:
            wind_strength = min(self.wind_condition.speed / 30.0, 1.0)
            course_map[5, :, :] = wind_strength
        
        return course_map

class ElitePlayer:
    def __init__(self, club_means, club_stds, lateral_stds, skill_level="tour_pro"):
        self.club_means = club_means
        self.club_stds = club_stds
        self.lateral_stds = lateral_stds
        self.skill_level = skill_level
        self.confidence = 0.5  # Start neutral
        self.momentum = 0.0
        
        # Skill level modifiers
        skill_modifiers = {
            "amateur": {"precision": 1.3, "power": 0.9, "consistency": 1.4},
            "club_pro": {"precision": 1.1, "power": 0.95, "consistency": 1.2},
            "tour_pro": {"precision": 1.0, "power": 1.0, "consistency": 1.0},
            "elite": {"precision": 0.85, "power": 1.05, "consistency": 0.8}
        }
        self.skill_mod = skill_modifiers.get(skill_level, skill_modifiers["tour_pro"])
        
    def simulate_elite_shot(self, club, aim, shot_shape, trajectory, tempo, 
                           mode, lie, wind_condition, pressure_level):
        """Enhanced shot simulation with multiple factors"""
        
        # Base shot parameters
        adj = mode_modifiers[mode]
        base_carry = self.club_means[club] + adj["carry_adj"]
        base_std_carry = self.club_stds[club] * adj["std_mult"]
        base_std_lateral = self.lateral_stds[club] * adj["std_mult"]
        
        # Apply skill level
        base_carry *= self.skill_mod["power"]
        base_std_carry *= self.skill_mod["consistency"]
        base_std_lateral *= self.skill_mod["precision"]
        
        # Lie adjustments
        lie_effects = {
            "Fairway": {"carry_mult": 1.0, "std_mult": 0.85},
            "First Cut": {"carry_mult": 0.98, "std_mult": 0.95},
            "Rough": {"carry_mult": 0.85, "std_mult": 1.15},
            "Deep Rough": {"carry_mult": 0.7, "std_mult": 1.3},
            "Bunker": {"carry_mult": 0.6, "std_mult": 1.4},
            "Tree": {"carry_mult": 0.5, "std_mult": 2.0},
            "Fringe": {"carry_mult": 0.95, "std_mult": 0.9}
        }
        
        lie_effect = lie_effects.get(lie, lie_effects["Fairway"])
        carry_multiplier = lie_effect["carry_mult"]
        std_multiplier = lie_effect["std_mult"]
        
        # Wind effects
        wind_carry_effect = 0
        wind_lateral_effect = 0
        
        if wind_condition.speed > 2:
            # Convert wind direction to carry/lateral components
            wind_radians = math.radians(wind_condition.direction)
            helping_component = math.cos(wind_radians) * wind_condition.speed * 0.8
            cross_component = math.sin(wind_radians) * wind_condition.speed * 0.6
            
            wind_carry_effect = helping_component
            wind_lateral_effect = cross_component
        
        # Shot shape effects
        shape_effects = {
            "straight": {"lateral_bias": 0, "carry_mult": 1.0},
            "draw": {"lateral_bias": -3, "carry_mult": 1.02},
            "fade": {"lateral_bias": 3, "carry_mult": 0.98}
        }
        shape_effect = shape_effects[shot_shape]
        
        # Trajectory effects
        traj_effects = {
            "low": {"carry_mult": 0.95, "wind_resistance": 0.7},
            "medium": {"carry_mult": 1.0, "wind_resistance": 1.0},
            "high": {"carry_mult": 1.03, "wind_resistance": 1.3}
        }
        traj_effect = traj_effects[trajectory]
        
        # Tempo effects
        tempo_effects = {
            "smooth": {"carry_mult": 0.97, "accuracy_mult": 0.8},
            "normal": {"carry_mult": 1.0, "accuracy_mult": 1.0},
            "aggressive": {"carry_mult": 1.05, "accuracy_mult": 1.3}
        }
        tempo_effect = tempo_effects[tempo]
        
        # Pressure effects
        pressure_penalty = pressure_level * 0.1
        std_multiplier *= (1 + pressure_penalty)
        
        # Confidence effects
        confidence_bonus = (self.confidence - 0.5) * 0.1
        carry_multiplier += confidence_bonus
        std_multiplier *= (1 - abs(confidence_bonus))
        
        # Strike quality
        strike_probs = [0.45, 0.25, 0.2, 0.1]  # Normal, Excellent, Slight mishit, Big mishit
        if pressure_level > 0.7:
            strike_probs = [0.35, 0.15, 0.3, 0.2]  # Worse under pressure
        
        strike = np.random.choice(
            ["Normal", "Excellent", "Slight Mishit", "Big Mishit"],
            p=strike_probs
        )
        
        strike_effects = {
            "Normal": {"carry_mult": 1.0, "std_mult": 1.0},
            "Excellent": {"carry_mult": 1.03, "std_mult": 0.3},
            "Slight Mishit": {"carry_mult": 0.92, "std_mult": 1.2},
            "Big Mishit": {"carry_mult": 0.78, "std_mult": 1.5}
        }
        strike_effect = strike_effects[strike]
        
        # Combine all effects
        final_carry_mult = (carry_multiplier * shape_effect["carry_mult"] * 
                           traj_effect["carry_mult"] * tempo_effect["carry_mult"] * 
                           strike_effect["carry_mult"])
        
        final_std_mult = (std_multiplier * tempo_effect["accuracy_mult"] * 
                         strike_effect["std_mult"])
        
        # Apply wind resistance to wind effects
        wind_carry_effect *= traj_effect["wind_resistance"]
        wind_lateral_effect *= traj_effect["wind_resistance"]
        
        # Generate shot
        final_carry = base_carry * final_carry_mult + wind_carry_effect
        final_std_carry = base_std_carry * final_std_mult
        final_std_lateral = base_std_lateral * final_std_mult
        
        carry = max(0, np.random.normal(final_carry, final_std_carry))
        lateral_deviation = (np.random.normal(0, final_std_lateral) + 
                           shape_effect["lateral_bias"] + wind_lateral_effect + aim)
        
        return ShotResult(
            carry=carry,
            lateral=lateral_deviation,
            final_position=(0, 0),  # Will be calculated by caller
            classification="",  # Will be set by caller
            strike_quality=strike,
            ball_flight=f"{trajectory}_{shot_shape}_{tempo}"
        )

# ====== ELITE REWARD SYSTEM ======
class EliteRewardCalculator:
    def __init__(self):
        self.strokes_gained_baseline = self._create_baseline_data()
        
    def _create_baseline_data(self):
        """Create strokes gained baseline data"""
        # Simplified strokes gained data (distance -> expected strokes)
        distances = list(range(0, 501, 10))
        expected_strokes = []
        
        for d in distances:
            if d == 0:
                expected_strokes.append(0)
            elif d <= 3:
                expected_strokes.append(1.0)
            elif d <= 10:
                expected_strokes.append(1.1)
            elif d <= 20:
                expected_strokes.append(1.3)
            elif d <= 50:
                expected_strokes.append(1.8)
            elif d <= 100:
                expected_strokes.append(2.4)
            elif d <= 150:
                expected_strokes.append(2.8)
            elif d <= 200:
                expected_strokes.append(3.1)
            elif d <= 250:
                expected_strokes.append(3.4)
            elif d <= 300:
                expected_strokes.append(3.6)
            elif d <= 400:
                expected_strokes.append(3.9)
            else:
                expected_strokes.append(4.2)
                
        return dict(zip(distances, expected_strokes))
    
    def calculate_strokes_gained(self, distance_before, distance_after, shot_result):
        """Calculate strokes gained for this shot"""
        def get_expected_strokes(distance):
            closest_dist = min(self.strokes_gained_baseline.keys(), 
                             key=lambda x: abs(x - distance))
            return self.strokes_gained_baseline[closest_dist]
        
        expected_before = get_expected_strokes(distance_before)
        expected_after = get_expected_strokes(distance_after)
        
        strokes_gained = expected_before - expected_after - 1.0  # -1 for the shot taken
        
        # Adjust for shot result quality
        if shot_result.classification == "OB":
            strokes_gained -= 2.0  # Penalty stroke + lost distance
        elif shot_result.classification == "Water":
            strokes_gained -= 1.5
        elif shot_result.classification == "Green":
            strokes_gained += 0.2  # Bonus for reaching green
            
        return strokes_gained
    
    def calculate_elite_reward(self, shot_context):
        """Multi-objective reward calculation"""
        # Unpack context
        distance_before = shot_context['distance_before']
        distance_after = shot_context['distance_after']
        shot_result = shot_context['shot_result']
        hole = shot_context['hole']
        game_context = shot_context['game_context']
        strategic_context = shot_context['strategic_context']
        
        # 1. Strokes Gained (40% of reward)
        sg_reward = self.calculate_strokes_gained(distance_before, distance_after, shot_result)
        
        # 2. Risk Management (25% of reward)
        risk_reward = self._calculate_risk_management_score(shot_result, strategic_context)
        
        # 3. Strategic Positioning (20% of reward)
        position_reward = self._calculate_positioning_score(shot_result, hole, distance_after)
        
        # 4. Pressure Performance (10% of reward)
        pressure_reward = self._calculate_pressure_performance(shot_result, game_context)
        
        # 5. Course Management (5% of reward)
        course_reward = self._calculate_course_management(shot_result, strategic_context)
        
        # Combine with weights
        total_reward = (0.4 * sg_reward + 
                       0.25 * risk_reward + 
                       0.2 * position_reward + 
                       0.1 * pressure_reward + 
                       0.05 * course_reward)
        
        return total_reward
    
    def _calculate_risk_management_score(self, shot_result, strategic_context):
        """Reward smart risk-taking"""
        base_score = 0
        
        # Reward avoiding trouble when conservative approach is wise
        if strategic_context['situation'] == 'leading' and shot_result.classification in ['Fairway', 'Green']:
            base_score += 2.0
        
        # Reward aggressive play when behind
        if strategic_context['situation'] == 'need_birdies' and shot_result.classification == 'Green':
            base_score += 3.0
            
        # Penalize unnecessary risks
        if strategic_context['situation'] == 'comfortable' and shot_result.classification in ['OB', 'Water']:
            base_score -= 4.0
            
        return base_score
    
    def _calculate_positioning_score(self, shot_result, hole, distance_after):
        """Reward shots that set up good next shots"""
        score = 0
        
        # Reward optimal approach distances
        if 80 <= distance_after <= 120:  # Sweet spot for approach shots
            score += 1.5
        elif 140 <= distance_after <= 160:  # Good mid-iron distance
            score += 1.0
            
        # Reward avoiding awkward distances
        if 40 <= distance_after <= 60:  # Awkward wedge distance
            score -= 1.0
            
        return score
    
    def _calculate_pressure_performance(self, shot_result, game_context):
        """Reward clutch performance"""
        if game_context.pressure_level < 0.3:
            return 0  # No pressure situation
            
        base_multiplier = game_context.pressure_level
        
        if shot_result.strike_quality == "Excellent":
            return 2.0 * base_multiplier
        elif shot_result.strike_quality == "Normal":
            return 1.0 * base_multiplier
        else:
            return -1.0 * base_multiplier
    
    def _calculate_course_management(self, shot_result, strategic_context):
        """Reward smart course management"""
        score = 0
        
        # Reward playing to strengths
        if strategic_context.get('playing_to_strength', False):
            score += 1.0
            
        # Reward wind management
        if strategic_context.get('wind_adjusted', False):
            score += 0.5
            
        return score

# ====== PRIORITIZED REPLAY BUFFER ======
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
    def push(self, state, action, reward, next_state, done, error=1.0):
        max_prio = self.priorities.max() if self.buffer else 1.0
        priority = (abs(error) + 1e-6) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)

# ====== ELITE TRAINING AGENT ======
class EliteGolfAgent:
    def __init__(self, learning_rate=ELITE_LEARNING_RATE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DuelingQNetwork().to(self.device)
        self.target_network = DuelingQNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Training components
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayBuffer(MEMORY_CAPACITY)
        self.reward_calculator = EliteRewardCalculator()
        
        # Training parameters
        self.batch_size = ELITE_BATCH_SIZE
        self.gamma = 0.95
        self.tau = 0.005  # Soft update parameter
        self.update_frequency = 4
        self.target_update_frequency = 100
        
        # Exploration
        self.epsilon_start = 0.9
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Learning state
        self.steps = 0
        self.episode = 0
        
        # LSTM hidden states
        self.lstm_hidden = None
        
    def get_action_space_size(self):
        """Calculate total action space size"""
        return len(clubs_all) * 21 * len(SHOT_SHAPES) * len(TRAJECTORIES) * len(SWING_TEMPOS)
    
    def encode_action(self, club, aim, shot_shape, trajectory, tempo):
        """Encode action into single integer"""
        club_idx = clubs_all.index(club)
        aim_idx = int((aim + 30) / 3)  # -30 to +30 in steps of 3 = 21 options
        shape_idx = SHOT_SHAPES.index(shot_shape)
        traj_idx = TRAJECTORIES.index(trajectory)
        tempo_idx = SWING_TEMPOS.index(tempo)
        
        action = (club_idx * 21 * 3 * 3 * 3 + 
                 aim_idx * 3 * 3 * 3 + 
                 shape_idx * 3 * 3 + 
                 traj_idx * 3 + 
                 tempo_idx)
        return action
    
    def decode_action(self, action):
        """Decode action integer back to components"""
        tempo_idx = action % 3
        action //= 3
        traj_idx = action % 3
        action //= 3
        shape_idx = action % 3
        action //= 3
        aim_idx = action % 21
        action //= 21
        club_idx = action
        
        club = clubs_all[club_idx]
        aim = aim_idx * 3 - 30  # Convert back to -30 to +30
        shot_shape = SHOT_SHAPES[shape_idx]
        trajectory = TRAJECTORIES[traj_idx]
        tempo = SWING_TEMPOS[tempo_idx]
        
        return club, aim, shot_shape, trajectory, tempo
    
    def create_state_tensor(self, hole, position, shot_history, game_context, wind_condition):
        """Create comprehensive state representation"""
        # Course map (6 channels)
        course_map = torch.FloatTensor(hole.create_2d_representation()).unsqueeze(0).to(self.device)
        
        # Shot history (last 5 shots encoded)
        history_tensor = self._encode_shot_history(shot_history)
        
        # Context vector
        context_vector = self._encode_context(position, game_context, wind_condition, hole)
        
        return course_map, history_tensor, context_vector
    
    def _encode_shot_history(self, shot_history):
        """Encode recent shot history for LSTM"""
        # Take last 5 shots, pad if necessary
        recent_shots = shot_history[-5:] if shot_history else []
        
        # Pad to exactly 5 shots
        while len(recent_shots) < 5:
            recent_shots.insert(0, self._empty_shot_encoding())
        
        # Encode each shot
        encoded_shots = []
        for shot in recent_shots:
            encoded = self._encode_single_shot(shot)
            encoded_shots.append(encoded)
        
        # Shape: (1, 5, 32) - batch_size=1, sequence_length=5, features=32
        return torch.FloatTensor(encoded_shots).unsqueeze(0).to(self.device)
    
    def _empty_shot_encoding(self):
        """Create encoding for missing shot (padding)"""
        return {
            'club': 'Driver', 'aim': 0, 'carry': 0, 'lateral': 0,
            'strike_quality': 'Normal', 'classification': 'Fairway',
            'distance_before': 0, 'distance_after': 0
        }
    
    def _encode_single_shot(self, shot):
        """Encode a single shot into feature vector"""
        # Club one-hot (14 features)
        club_encoding = [1 if shot['club'] == club else 0 for club in clubs_all]
        
        # Normalized continuous features (18 features)
        features = [
            shot['aim'] / 30.0,  # Normalize aim
            shot.get('carry', 0) / 300.0,  # Normalize carry
            shot.get('lateral', 0) / 50.0,  # Normalize lateral
            shot.get('distance_before', 0) / 500.0,  # Normalize distance
            shot.get('distance_after', 0) / 500.0,  # Normalize distance
        ]
        
        # Strike quality encoding (4 features)
        strike_qualities = ['Normal', 'Excellent', 'Slight Mishit', 'Big Mishit']
        strike_encoding = [1 if shot.get('strike_quality', 'Normal') == sq else 0 for sq in strike_qualities]
        
        # Classification encoding (9 features) 
        classifications = ['Fairway', 'Rough', 'Deep Rough', 'Bunker', 'Tree', 'Green', 'Fringe', 'OB', 'Water']
        class_encoding = [1 if shot.get('classification', 'Fairway') == cls else 0 for cls in classifications]
        
        return club_encoding + features + strike_encoding + class_encoding  # Total: 32 features
    
    def _encode_context(self, position, game_context, wind_condition, hole):
        """Encode contextual information"""
        distance, lateral = position
        
        context = [
            # Position (2 features)
            distance / 500.0,
            lateral / 50.0,
            
            # Hole characteristics (6 features)
            hole.par / 5.0,
            hole.green_distance / 500.0,
            hole.fairway_width / 50.0,
            hole.green_width / 30.0,
            hole.green_depth / 40.0,
            hole.elevation / 30.0,
            
            # Wind (3 features)
            wind_condition.speed / 30.0,
            math.cos(math.radians(wind_condition.direction)),
            math.sin(math.radians(wind_condition.direction)),
            
            # Game context (9 features)
            game_context.round_number / 4.0,
            game_context.hole_number / 18.0,
            (game_context.current_score_vs_par + 10) / 20.0,  # Normalize around typical range
            game_context.pressure_level,
            game_context.confidence,
            game_context.momentum,
        ]
        
        # Tournament position encoding (4 features)
        positions = ['leading', 'contending', 'making_cut', 'missing_cut']
        pos_encoding = [1 if game_context.tournament_position == pos else 0 for pos in positions]
        context.extend(pos_encoding)
        
        return torch.FloatTensor(context).unsqueeze(0).to(self.device)  # Total: 20 features
    
    def select_action(self, state_tensors, available_clubs, exploration=True):
        """Select action using epsilon-greedy with neural network"""
        course_map, shot_history, context = state_tensors
        
        if exploration and random.random() < self.epsilon:
            # Random action
            club = random.choice(available_clubs)
            aim = random.choice(range(-30, 31, 3))
            shot_shape = random.choice(SHOT_SHAPES)
            trajectory = random.choice(TRAJECTORIES)
            tempo = random.choice(SWING_TEMPOS)
            
            return club, aim, shot_shape, trajectory, tempo
        
        # Neural network action selection
        with torch.no_grad():
            q_values, self.lstm_hidden = self.q_network(course_map, shot_history, context, self.lstm_hidden)
            
            # Mask unavailable clubs
            available_club_indices = [clubs_all.index(club) for club in available_clubs]
            
            # Find best action among available clubs
            best_action = None
            best_q_value = float('-inf')
            
            for club in available_clubs:
                for aim in range(-30, 31, 3):
                    for shot_shape in SHOT_SHAPES:
                        for trajectory in TRAJECTORIES:
                            for tempo in SWING_TEMPOS:
                                action_idx = self.encode_action(club, aim, shot_shape, trajectory, tempo)
                                
                                if action_idx < q_values.size(1):
                                    q_val = q_values[0, action_idx].item()
                                    if q_val > best_q_value:
                                        best_q_value = q_val
                                        best_action = (club, aim, shot_shape, trajectory, tempo)
            
            if best_action is None:
                # Fallback to random if no valid action found
                club = random.choice(available_clubs)
                aim = random.choice(range(-30, 31, 3))
                shot_shape = random.choice(SHOT_SHAPES) 
                trajectory = random.choice(TRAJECTORIES)
                tempo = random.choice(SWING_TEMPOS)
                return club, aim, shot_shape, trajectory, tempo
                
            return best_action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in prioritized replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss with importance sampling weights
        td_errors = current_q_values.squeeze() - target_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Soft update target network
        if self.steps % self.target_update_frequency == 0:
            self._soft_update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
        
        return loss.item()
    
    def _soft_update_target_network(self):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# ====== ELITE GAME SIMULATION ======
def classify_shot_elite(hole, x, y):
    """Enhanced shot classification"""
    # Green boundaries
    green_x_start = hole.green_distance - hole.green_depth/2
    green_x_end = hole.green_distance + hole.green_depth/2
    green_y_start = -hole.green_width/2
    green_y_end = hole.green_width/2
    
    # Check OB first
    for ob in hole.ob_zones:
        if ob["x_start"] <= x <= ob["x_end"] and ob["y_start"] <= y <= ob["y_end"]:
            return "OB"
    
    # Check water
    for wz in hole.water_zones:
        if wz["x_start"] <= x <= wz["x_end"] and wz["y_start"] <= y <= wz["y_end"]:
            return "Water"
    
    # Check other zones
    for zone in hole.zones:
        if zone["x_start"] <= x <= zone["x_end"] and zone["y_start"] <= y <= zone["y_end"]:
            if zone["type"] == "Fringe":
                if green_x_start <= x <= green_x_end and green_y_start <= y <= green_y_end:
                    return "Green"
                else:
                    return "Fringe"
            return zone["type"]
    
    # Green check
    if green_x_start <= x <= green_x_end and green_y_start <= y <= green_y_end:
        return "Green"
    
    # Fringe around green
    fringe_buffer = 6
    if ((green_x_start - fringe_buffer) <= x <= (green_x_end + fringe_buffer) and 
        (green_y_start - fringe_buffer) <= y <= (green_y_end + fringe_buffer)):
        return "Fringe"
    
    # Fairway and rough
    if abs(y) <= hole.fairway_width/2:
        return "Fairway"
    elif abs(y) <= hole.fairway_width/2 + 6:
        return "First Cut"
    elif abs(y) <= hole.fairway_width/2 + 20:
        return "Rough"
    else:
        return "Deep Rough"

def generate_elite_hole():
    """Generate a challenging, realistic hole"""
    # More varied hole types
    hole_types = ["dogleg_right", "dogleg_left", "straight", "risk_reward"]
    hole_type = random.choice(hole_types)
    
    elevation = np.random.uniform(-40, 40)
    elevation_yards = elevation / 3
    
    if hole_type == "risk_reward":
        base_distance = np.random.uniform(500, 550)  # Par 5
        par = 5
    else:
        base_distance = np.random.uniform(380, 480)
        par = 4 if base_distance < 450 else 5
    
    green_distance = base_distance + elevation_yards
    fairway_width = np.random.uniform(22, 42)
    green_depth = np.random.uniform(20, 35)
    green_width = np.random.uniform(16, 28)
    
    # Pin position with more variety
    pin_difficulty = random.choice(["easy", "medium", "hard"])
    if pin_difficulty == "easy":
        pin_x = green_distance - green_depth/2 + green_depth * random.uniform(0.3, 0.7)
        pin_y = -green_width/2 + green_width * random.uniform(0.3, 0.7)
    elif pin_difficulty == "medium":
        pin_x = green_distance - green_depth/2 + green_depth * random.uniform(0.2, 0.8)
        pin_y = -green_width/2 + green_width * random.uniform(0.2, 0.8)
    else:  # hard
        # Pin close to edges
        if random.random() < 0.5:
            pin_x = green_distance - green_depth/2 + green_depth * random.choice([0.1, 0.9])
        else:
            pin_x = green_distance - green_depth/2 + green_depth * random.uniform(0.3, 0.7)
        pin_y = -green_width/2 + green_width * random.choice([0.1, 0.9])
    
    pin_pos = (pin_x, pin_y)
    
    # Enhanced hazard placement
    zones = []
    
    # Strategic bunkers
    num_bunkers = random.randint(1, 4)
    for _ in range(num_bunkers):
        # Place bunkers at strategic distances
        strategic_distances = [150, 200, 250, green_distance - 50]
        bunker_x = random.choice(strategic_distances) + random.uniform(-30, 30)
        bunker_width = random.uniform(15, 35)
        bunker_y_center = random.uniform(-25, 25)
        
        zones.append({
            "type": "Bunker",
            "x_start": bunker_x,
            "x_end": bunker_x + bunker_width,
            "y_start": bunker_y_center - 8,
            "y_end": bunker_y_center + 8
        })
    
    # Water hazards for risk-reward holes
    if hole_type == "risk_reward" or random.random() < 0.4:
        water_x = random.uniform(200, green_distance - 80)
        water_width = random.uniform(30, 60)
        water_y_center = random.uniform(-20, 20)
        
        zones.append({
            "type": "Water",
            "x_start": water_x,
            "x_end": water_x + water_width,
            "y_start": water_y_center - 15,
            "y_end": water_y_center + 15
        })
    
    # Trees for doglegs
    if "dogleg" in hole_type:
        tree_side = 1 if "right" in hole_type else -1
        tree_x = random.uniform(180, 280)
        
        zones.append({
            "type": "Tree",
            "x_start": tree_x,
            "x_end": tree_x + 40,
            "y_start": tree_side * 20,
            "y_end": tree_side * 45
        })
    
    # Fringe around green
    zones.append({
        "type": "Fringe",
        "x_start": green_distance - green_depth/2 - 6,
        "x_end": green_distance + green_depth/2 + 6,
        "y_start": -green_width/2 - 6,
        "y_end": green_width/2 + 6
    })
    
    # OB zones
    ob_zones = [
        {"x_start": 0, "x_end": green_distance + 20, "y_start": 50, "y_end": 120, "type": "OB"},
        {"x_start": 0, "x_end": green_distance + 20, "y_start": -120, "y_end": -50, "type": "OB"}
    ]
    
    # Generate wind condition
    wind_speed = random.uniform(0, 25)
    wind_direction = random.uniform(0, 360)
    wind_gusts = random.uniform(0, wind_speed * 0.3)
    wind_condition = WindCondition(wind_speed, wind_direction, wind_gusts)
    
    return EliteGolfHole(
        length=base_distance,
        fairway_width=fairway_width,
        green_distance=green_distance,
        green_depth=green_depth,
        green_width=green_width,
        elevation=elevation,
        zones=zones,
        par=par,
        pin_pos=pin_pos,
        ob_zones=ob_zones,
        water_zones=[z for z in zones if z.get("type") == "Water"],
        wind_condition=wind_condition
    )

def elite_putting_simulation(distance_to_pin, zone, pressure_level, confidence):
    """Enhanced putting with psychological factors"""
    # Base probabilities adjusted for pressure and confidence
    pressure_penalty = pressure_level * 0.1
    confidence_bonus = (confidence - 0.5) * 0.05
    
    miss_factor = 1.0
    if zone == "Fringe":
        miss_factor = 1.3
    
    # Apply psychological adjustments
    miss_factor *= (1 + pressure_penalty - confidence_bonus)
    
    if distance_to_pin <= 3:
        make_prob = (0.95 if zone == "Green" else 0.85) / miss_factor
        if np.random.rand() < make_prob:
            return 1
        else:
            return 2
    elif distance_to_pin <= 8:
        make_prob = (0.55 if zone == "Green" else 0.40) / miss_factor
        if np.random.rand() < make_prob:
            return 1
        else:
            return 2
    elif distance_to_pin <= 15:
        if np.random.rand() < (0.15 / miss_factor):
            return 1
        elif np.random.rand() < (0.85 / miss_factor):
            return 2
        else:
            return 3
    elif distance_to_pin <= 30:
        if np.random.rand() < (0.05 / miss_factor):
            return 1
        elif np.random.rand() < (0.80 / miss_factor):
            return 2
        else:
            return 3
    else:
        if np.random.rand() < (0.08 / miss_factor):
            return 2
        elif np.random.rand() < (0.85 / miss_factor):
            return 3
        else:
            return 4

def run_elite_episode(hole, agent, player, game_context):
    """Run a single elite episode with comprehensive state tracking"""
    # Initialize episode state
    distance = hole.green_distance
    lateral = 0.0
    shot_num = 0
    done = False
    total_strokes = 0
    shot_history = []
    
    # Game state
    lie = "Fairway"
    pin_x, pin_y = hole.pin_pos
    
    # Safety counter
    safety_counter = 0
    max_shots = MAX_SHOTS_PER_HOLE
    
    # Reset LSTM hidden state for new episode
    agent.lstm_hidden = None
    
    while not done and safety_counter < max_shots:
        safety_counter += 1
        shot_num += 1
        
        # Determine available clubs
        if shot_num == 1:
            available_clubs = tee_clubs
        else:
            available_clubs = [c for c in clubs_all if c != "Driver"]
        
        # Create state representation
        position = (distance, lateral)
        state_tensors = agent.create_state_tensor(
            hole, position, shot_history, game_context, hole.wind_condition
        )
        
        # Select action
        club, aim, shot_shape, trajectory, tempo = agent.select_action(
            state_tensors, available_clubs, exploration=True
        )
        
        # Determine mode based on game context and recent performance
        mode = determine_shot_mode(game_context, shot_history, distance)
        
        # Simulate shot
        shot_result = player.simulate_elite_shot(
            club, aim, shot_shape, trajectory, tempo, mode, lie, 
            hole.wind_condition, game_context.pressure_level
        )
        
        # Update position
        new_distance = max(0, distance - shot_result.carry)
        new_lateral = lateral + shot_result.lateral
        
        # Determine ball position and classification
        ball_x = hole.green_distance - new_distance
        ball_y = new_lateral
        classification = classify_shot_elite(hole, ball_x, ball_y)
        
        # Update shot result
        shot_result.final_position = (ball_x, ball_y)
        shot_result.classification = classification
        
        # Calculate pin distance
        pin_dist = np.sqrt((ball_x - pin_x) ** 2 + (ball_y - pin_y) ** 2)
        
        # Handle special cases
        if classification in ["OB", "Water"]:
            # Penalty shot - return to previous position with penalty
            total_strokes += 2  # Stroke + penalty
            distance = max(distance, 100)  # Don't go backwards past reasonable point
            lateral = 0  # Reset lateral to safe position
            lie = "Fairway"
            
            # Add penalty to shot history
            shot_info = {
                'shot_num': shot_num,
                'club': club,
                'aim': aim,
                'shot_shape': shot_shape,
                'trajectory': trajectory,
                'tempo': tempo,
                'carry': shot_result.carry,
                'lateral': shot_result.lateral,
                'strike_quality': shot_result.strike_quality,
                'classification': classification,
                'distance_before': distance,
                'distance_after': distance,  # No progress
                'pin_distance': pin_dist,
                'mode': mode
            }
            shot_history.append(shot_info)
            
            continue
        
        # Update position and lie
        distance = new_distance
        lateral = new_lateral
        lie = classification
        total_strokes += 1
        
        # Check for putting
        if classification in ["Green", "Fringe"] and distance <= 40:
            # Putting phase
            putts = elite_putting_simulation(
                pin_dist, classification, game_context.pressure_level, game_context.confidence
            )
            total_strokes += putts
            done = True
            
            # Update final shot info
            shot_info = {
                'shot_num': shot_num,
                'club': club,
                'aim': aim,
                'shot_shape': shot_shape,
                'trajectory': trajectory,
                'tempo': tempo,
                'carry': shot_result.carry,
                'lateral': shot_result.lateral,
                'strike_quality': shot_result.strike_quality,
                'classification': classification,
                'distance_before': distance + shot_result.carry,
                'distance_after': 0,  # Holed out
                'pin_distance': pin_dist,
                'mode': mode,
                'putts': putts
            }
            
        else:
            # Regular shot
            shot_info = {
                'shot_num': shot_num,
                'club': club,
                'aim': aim,
                'shot_shape': shot_shape,
                'trajectory': trajectory,
                'tempo': tempo,
                'carry': shot_result.carry,
                'lateral': shot_result.lateral,
                'strike_quality': shot_result.strike_quality,
                'classification': classification,
                'distance_before': distance + shot_result.carry,
                'distance_after': distance,
                'pin_distance': pin_dist,
                'mode': mode
            }
        
        shot_history.append(shot_info)
        
        # Calculate reward for this shot
        shot_context = {
            'distance_before': shot_info['distance_before'],
            'distance_after': shot_info['distance_after'],
            'shot_result': shot_result,
            'hole': hole,
            'game_context': game_context,
            'strategic_context': determine_strategic_context(game_context, hole, distance, shot_history)
        }
        
        reward = agent.reward_calculator.calculate_elite_reward(shot_context)
        shot_info['reward'] = reward
        
        # Update player confidence based on shot result
        update_player_psychology(player, shot_result, game_context)
    
    # Calculate final score
    score_vs_par = total_strokes - hole.par
    
    # Create result summary
    result_summary = {
        'total_strokes': total_strokes,
        'score_vs_par': score_vs_par,
        'shots_to_green': len([s for s in shot_history if s['classification'] not in ['Green', 'Fringe']]),
        'fairways_hit': len([s for s in shot_history if s['classification'] == 'Fairway']),
        'greens_in_regulation': any(s['classification'] == 'Green' for s in shot_history[:-1]),
        'total_putts': sum(s.get('putts', 0) for s in shot_history),
        'shot_history': shot_history
    }
    
    return result_summary

def determine_shot_mode(game_context, shot_history, distance):
    """Determine shot mode based on game context and recent performance"""
    base_mode = "Normal"
    
    # Adjust based on pressure
    if game_context.pressure_level > 0.7:
        base_mode = "Pressure"
    elif game_context.confidence > 0.7:
        base_mode = "Good"
    elif game_context.confidence > 0.8:
        base_mode = "VeryGood"
    elif game_context.confidence < 0.3:
        base_mode = "Bad"
    
    # Adjust based on recent performance
    if shot_history:
        recent_shots = shot_history[-3:]  # Last 3 shots
        good_shots = sum(1 for s in recent_shots if s.get('strike_quality') in ['Excellent', 'Normal'])
        
        if good_shots == len(recent_shots) and len(recent_shots) >= 2:
            # On a roll
            if base_mode == "Normal":
                base_mode = "Good"
            elif base_mode == "Good":
                base_mode = "VeryGood"
        elif good_shots == 0 and len(recent_shots) >= 2:
            # Struggling
            base_mode = "Bad"
    
    return base_mode

def determine_strategic_context(game_context, hole, distance, shot_history):
    """Determine strategic context for reward calculation"""
    context = {}
    
    # Determine situation
    if game_context.current_score_vs_par <= -2:
        context['situation'] = 'leading'
    elif game_context.current_score_vs_par >= 2:
        context['situation'] = 'need_birdies'
    else:
        context['situation'] = 'comfortable'
    
    # Wind adjustment check
    if hole.wind_condition.speed > 10:
        context['wind_adjusted'] = True
    else:
        context['wind_adjusted'] = False
    
    # Playing to strength (simplified)
    if distance < 150 and len([c for c in shot_history if c.get('club') in wedge_names]) > 0:
        context['playing_to_strength'] = True
    else: