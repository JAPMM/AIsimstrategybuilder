import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import Counter, deque
import csv
import os

# ====== CLUB DATA LOAD ======
summary_df = pd.read_csv("golf_shot_dispersion_summary.csv")
clubs_all = summary_df["Club"].tolist()
club_means = dict(zip(summary_df["Club"], summary_df["mean_carry"]))
club_stds = dict(zip(summary_df["Club"], summary_df["std_carry"]))
lateral_stds = dict(zip(summary_df["Club"], summary_df["std_lateral"]))
tee_clubs = [c for c in clubs_all if club_means[c] >= 180]
wedge_names = ["Pitching Wedge","50* Wedge","54* Wedge","62* Wedge"]

mode_modifiers = {
    "VeryGood": {"carry_adj": +15, "std_mult": 0.8},
    "Good": {"carry_adj": +7, "std_mult": 0.9},
    "Normal": {"carry_adj": 0, "std_mult": 1.0},
    "Bad": {"carry_adj": -10, "std_mult": 1.3}
}
mode_encoding_map = {
    "VeryGood": 0,
    "Good": 1,
    "Normal": 2,
    "Bad": 3,
    "Mixed": 4
}
mode_list = ["VeryGood", "Good", "Normal", "Bad", "Mixed"]

# ====== ENVIRONMENT/HANDLER CLASSES ======
class GolfHole:
    def __init__(self, length, fairway_width, green_distance, green_depth, green_width, elevation, zones, par, pin_pos, ob_zones=None, water_zones=None):
        self.length = length
        self.fairway_width = fairway_width
        self.green_distance = green_distance
        self.green_depth = green_depth
        self.green_width = green_width
        self.elevation = elevation
        self.zones = zones
        self.par = par
        self.pin_pos = pin_pos
        self.ob_zones = ob_zones if ob_zones else []
        self.water_zones = water_zones if water_zones else []

def generate_random_hole():
    elevation = np.random.uniform(-30, 30)
    elevation_yards = elevation / 3
    base_distance = np.random.uniform(420, 510)
    green_distance = base_distance + elevation_yards
    fairway_width = np.random.uniform(25, 38)
    green_depth = np.random.uniform(22, 32)
    green_width = np.random.uniform(18, 26)
    par = 4 if green_distance < 450 else 5
    pin_x = green_distance - green_depth/2 + np.random.uniform(0, green_depth)
    pin_y = -green_width/2 + np.random.uniform(0, green_width)
    pin_pos = (pin_x, pin_y)
    ob_zones = [
        {"x_start": 0, "x_end": green_distance+10, "y_start": 40, "y_end": 100, "type":"OB"},
        {"x_start": 0, "x_end": green_distance+10, "y_start": -100, "y_end": -40, "type":"OB"}
    ]
    water_zones = []
    if np.random.rand() < 0.3:
        wx = np.random.uniform(150, green_distance-100)
        water_zones.append({"x_start": wx, "x_end": wx+35, "y_start": -24, "y_end": 24, "type":"Water"})
    num_bunkers = np.random.choice([1, 2])
    bunkers = []
    for _ in range(num_bunkers):
        bx = np.random.uniform(180, green_distance-70)
        bunkers.append({"type":"Bunker",
                        "x_start": bx,
                        "x_end": bx+30,
                        "y_start": np.random.uniform(-15, 5),
                        "y_end": np.random.uniform(10, 25)})
    zones = bunkers + water_zones
    if np.random.rand() < 0.7:
        zones.append({"type":"Tree",
                      "x_start": 0,
                      "x_end": green_distance,
                      "y_start": np.random.uniform(25, 35),
                      "y_end": np.random.uniform(38, 50)})
    zones.append({
        "type":"Fringe",
        "x_start": green_distance - green_depth/2 - 6,
        "x_end": green_distance + green_depth/2 + 6,
        "y_start": -green_width/2 - 6,
        "y_end": green_width/2 + 6
    })
    return GolfHole(
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
        water_zones=water_zones
    )

class Player:
    def __init__(self, club_means, club_stds, lateral_stds):
        self.club_means = club_means
        self.club_stds = club_stds
        self.lateral_stds = lateral_stds
    def simulate_shot(self, club, aim, mode, lie):
        adj = mode_modifiers[mode]
        mean = self.club_means[club] + adj["carry_adj"]
        base_std_carry = self.club_stds[club] * adj["std_mult"]
        base_std_lat = self.lateral_stds[club] * adj["std_mult"]
        if lie == "Fairway":
            carry_mult = 1.0
            std_mult = np.random.uniform(0.8, 0.88)
        elif lie == "Rough":
            carry_mult = np.random.uniform(0.8, 0.9)
            std_mult = np.random.uniform(1.03, 1.09)
        elif lie == "Deep Rough":
            carry_mult = np.random.uniform(0.65, 0.8)
            std_mult = np.random.uniform(1.12, 1.22)
        elif lie == "Bunker":
            carry_mult = np.random.uniform(0.5, 0.7)
            std_mult = np.random.uniform(1.15, 1.25)
        elif lie == "Tree":
            carry_mult = np.random.uniform(0.4, 0.6)
            std_mult = 2.0
        elif lie == "Fringe":
            carry_mult = 0.95
            std_mult = 0.93
        else:
            carry_mult = 1.0
            std_mult = 1.0
        strike = np.random.choice(
            ["Normal", "Excellent", "Slight Mishit", "Big Mishit"],
            p=[0.5, 0.2, 0.2, 0.1]
        )
        if strike == "Normal":
            pass
        elif strike == "Excellent":
            carry_mult *= np.random.uniform(1.0, 1.05)
            std_mult = 0.01
        elif strike == "Slight Mishit":
            carry_mult *= 0.9
            std_mult *= 1.05
        elif strike == "Big Mishit":
            carry_mult *= np.random.uniform(0.75, 0.8)
            std_mult *= 1.10
        std_carry = base_std_carry * std_mult
        std_lat = base_std_lat * std_mult
        carry = np.random.normal(mean * carry_mult, std_carry)
        lateral = np.random.normal(0, std_lat) + aim
        return max(0, carry), lateral, strike

class QNetwork(nn.Module):
    def __init__(self, input_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def classify_shot(hole, x, y):
    green_x_start = hole.green_distance - hole.green_depth/2
    green_x_end = hole.green_distance + hole.green_depth/2
    green_y_start = -hole.green_width/2
    green_y_end = hole.green_width/2
    for ob in hole.ob_zones:
        if ob["x_start"] <= x <= ob["x_end"] and ob["y_start"] <= y <= ob["y_end"]:
            return "OB"
    for wz in hole.water_zones:
        if wz["x_start"] <= x <= wz["x_end"] and wz["y_start"] <= y <= wz["y_end"]:
            return "Water"
    for zone in hole.zones:
        if zone["x_start"] <= x <= zone["x_end"] and zone["y_start"] <= y <= zone["y_end"]:
            if zone["type"] == "Fringe":
                if green_x_start <= x <= green_x_end and green_y_start <= y <= green_y_end:
                    return "Green"
                else:
                    return "Fringe"
            return zone["type"]
    if green_x_start <= x <= green_x_end and green_y_start <= y <= green_y_end:
        return "Green"
    if (green_x_start-6) <= x <= (green_x_end+6) and (green_y_start-6) <= y <= (green_y_end+6):
        return "Fringe"
    if abs(y) <= hole.fairway_width/2:
        return "Fairway"
    elif abs(y) <= hole.fairway_width/2 + 6:
        return "First Cut"
    elif abs(y) <= hole.fairway_width/2 + 20:
        return "Rough"
    else:
        return "Deep Rough"

def reward_for_shot(classification, remaining, lateral, shot_num, prev_distance, new_distance, club, mode_sample, hole, pin_dist, putt_attempted, last_lateral, shot_log, streak_score, same_club_count):
    reward = -1
    if abs(lateral) < 5:
        reward += 3.0
    elif abs(lateral) < 10:
        reward += 1.0
    elif abs(lateral) < 20:
        reward -= 1.0
    elif abs(lateral) < 40:
        reward -= 2.5
    else:
        reward -= 6.0
    if last_lateral is not None:
        if abs(lateral) > abs(last_lateral):
            reward -= 3.0
        elif abs(lateral) < abs(last_lateral):
            reward += 1.2
    if classification == "OB":
        reward -= 8.0
    if classification == "Water":
        reward -= 6.0
    if shot_num > 1 and shot_log and shot_log[-1]['lie'] in ["Rough","Deep Rough","Tree","Bunker"] and classification in ["Fairway","Fringe","Green"]:
        reward += 2.5
    if classification == "Tree" and (abs(lateral) > 25):
        reward -= 3.0
    if club in wedge_names and remaining >= 80:
        reward -= 3.0
    if classification == "Green" and club in wedge_names:
        reward += 1.0
    if remaining<=20 and club in wedge_names:
        reward += 2.0
    if putt_attempted:
        if classification not in ["Green", "Fringe"]:
            reward -= 3.0
        elif pin_dist < 6:
            reward += 2.0
        elif pin_dist < 12:
            reward += 1.3
        elif pin_dist < 24:
            reward += 0.6
        elif pin_dist > 50:
            reward -= 2.0
    if shot_num==1 and (80 <= remaining <=140):
        reward +=2.5
    if 45 <= remaining <= 65:
        reward -=2.5
    if streak_score > 0:
        reward += 0.1 * streak_score
    elif streak_score < 0:
        reward += 0.2 * streak_score
    if same_club_count >= 3:
        reward -= 2.5
    if classification in ["Bunker","Tree","Deep Rough"]:
        reward -=1.5
    elif classification=="Rough":
        reward -=0.7
    elif classification=="First Cut":
        reward -=0.4
    reward += 0.1 * (prev_distance - new_distance)
    reward -=0.005 * remaining
    if (prev_distance - new_distance) <=15:
        reward -=1.2
    carry_diff = abs(club_means[club]-remaining)
    if carry_diff <=10:
        reward +=1.0
    elif carry_diff <=20:
        reward +=0.5
    elif carry_diff >=30 and remaining < 160:
        reward -=1.0
    if shot_num==1:
        reward +=0.7
    if classification=="Green":
        reward +=1.0
    if remaining==0 and classification!="Green":
        reward -=1.0
    if new_distance<0:
        reward -=0.5
    if remaining<=50:
        reward +=0.5
    if remaining<=20:
        reward +=0.75
    if remaining<=100:
        if club in wedge_names:
            reward +=0.5
        wedge_distances = {w: abs(club_means[w]-remaining) for w in wedge_names}
        best_wedge = min(wedge_distances, key=wedge_distances.get)
        if club==best_wedge:
            reward +=1.0
    if mode_sample=="VeryGood" and club in ["Driver","3 Wood"]:
        reward +=0.3
    if mode_sample=="Bad" and club in ["Driver","3 Wood"]:
        reward -=0.5
    if mode_sample=="Bad" and remaining<=100 and club in wedge_names:
        reward +=0.5
    reward -= club_distance_penalty(club, remaining)
    return reward

def club_distance_penalty(club, remaining):
    mean_carry = club_means[club]
    if abs(mean_carry - remaining)<20:
        return 0
    return 0.12 * abs(mean_carry - remaining)

def putting_strokes(distance_to_pin, zone):
    miss_factor = 1.0
    if zone == "Fringe":
        miss_factor = 1.5
    if distance_to_pin <= 3:
        make_prob = 0.97 if zone=="Green" else 0.88
        if np.random.rand() < make_prob:
            return 1
        else:
            return 2
    elif distance_to_pin <= 10:
        make_prob = 0.45 if zone=="Green" else 0.32
        if np.random.rand() < make_prob:
            return 1
        else:
            return 2
    elif distance_to_pin <= 20:
        if np.random.rand() < (0.08/miss_factor):
            return 1
        elif np.random.rand() < (0.90/miss_factor):
            return 2
        else:
            return 3
    elif distance_to_pin <= 40:
        if np.random.rand() < (0.02/miss_factor):
            return 1
        elif np.random.rand() < (0.82/miss_factor):
            return 2
        else:
            return 3
    else:
        if np.random.rand() < (0.06/miss_factor):
            return 2
        else:
            return 3

def encode_state_action(hole_idx, distance, lateral, shot_num, mode_encoding, club, aim):
    club_onehot = [1 if club==c else 0 for c in clubs_all]
    aim_norm = aim/15
    return [hole_idx, distance/500, lateral/50, shot_num/10, mode_encoding/4]+club_onehot+[aim_norm]

# ------------------- REPLAY BUFFER CLASS ------------------
class ReplayBuffer:
    def __init__(self, capacity=250000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return np.stack(state), np.stack(action), np.stack(reward), np.stack(next_state), np.stack(done)
    def __len__(self):
        return len(self.buffer)

# ---------------- SHOT LOG REPLAY LOADER -----------------
def load_shotlog_to_replay(filename, encode_state_action_func, filter_invalid=True):
    replay = ReplayBuffer()
    if not os.path.exists(filename):
        return replay
    df = pd.read_csv(filename)
    episodes = df["ep"].unique()
    for ep in episodes:
        ep_df = df[df["ep"] == ep].reset_index(drop=True)
        # Filter out invalid episodes
        if filter_invalid and any(str(res).lower().find("stuck in loop") != -1 for res in ep_df["classification"].values):
            continue
        for i in range(len(ep_df)-1):
            s = ep_df.iloc[i]
            s_next = ep_df.iloc[i+1]
            state = encode_state_action_func(
                0,
                s["distance_to_pin"],
                s["lateral"],
                s["shot_num"],
                mode_encoding_map.get(s.get("mode","Normal"),2),
                s["club"],
                s["aim"]
            )
            action = [clubs_all.index(s["club"]), s["aim"]/15]
            reward = [s["reward"]]
            next_state = encode_state_action_func(
                0,
                s_next["distance_to_pin"],
                s_next["lateral"],
                s_next["shot_num"],
                mode_encoding_map.get(s_next.get("mode","Normal"),2),
                s_next["club"],
                s_next["aim"]
            )
            done = [int(i+1 == len(ep_df)-1)]
            replay.push(state, action, reward, next_state, done)
    print(f"Replay buffer loaded: {len(replay)} transitions from {filename}")
    return replay

# ======= SINGLE EPISODE RUNNER =======
def run_single_episode(hole, qnet, player, clubs_all, tee_clubs, epsilon, mode_list):
    distance = hole.green_distance
    lateral = 0
    done = False
    shot_num = 1
    mode = random.choice(mode_list)
    total_reward = 0
    total_strokes = 0
    lie = "Fairway"
    pin_x, pin_y = hole.pin_pos
    safety_counter = 0
    last_lateral = 0
    shot_log = []
    club_list = []
    same_club_count = 0
    last_club = None
    score_streak = 0
    while not done:
        safety_counter += 1
        if safety_counter > 20:
            result = "stuck in loop"
            done = True
            break
        if shot_num == 1:
            available_clubs = tee_clubs
        else:
            available_clubs = [c for c in clubs_all if c != "Driver"]
        if mode == "Mixed":
            mode_sample = random.choices(
                ["VeryGood", "Good", "Normal", "Bad"],
                weights=[48, 12, 15, 25],
                k=1
            )[0]
        else:
            mode_sample = mode
        mode_enc = mode_encoding_map[mode_sample]
        if club_list and last_club is not None and last_club == club_list[-1]:
            same_club_count += 1
        else:
            same_club_count = 1
        if random.uniform(0, 1) < epsilon:
            club = random.choice(available_clubs)
            aim = random.choice([-15,0,15])
        else:
            qvals = []
            for c in available_clubs:
                for a in [-15,0,15]:
                    sa = torch.tensor(encode_state_action(0, distance, lateral, shot_num, mode_enc, c, a), dtype=torch.float32)
                    qvals.append(qnet(sa).item())
            idx = np.argmax(qvals)
            club, aim = [(c, a) for c in available_clubs for a in [-15,0,15]][idx]
        club_list.append(club)
        last_club = club
        carry, y_offset, strike_quality = player.simulate_shot(club, aim, mode_sample, lie)
        new_distance = max(0, distance - carry)
        new_lateral = lateral + y_offset
        ball_x = hole.green_distance - new_distance
        ball_y = new_lateral
        classification = classify_shot(hole, ball_x, ball_y)
        pin_dist = np.sqrt((ball_x - pin_x) ** 2 + (ball_y - pin_y) ** 2)
        putt_attempted = (classification in ["Green", "Fringe"]) and new_distance <= 40
        green_fringe_limit = (hole.green_width / 2 + 6)
        if new_distance <= 0 and abs(new_lateral) <= green_fringe_limit and classification not in ["Green","Fringe"]:
            distance = 0
            lateral = new_lateral
            shot_num += 1
            lie = "Rough"
            total_reward -= 1.0
            shot_log.append({
                "shot_num": shot_num,
                "club": club,
                "aim": aim,
                "carry": carry,
                "lateral": y_offset,
                "strike": strike_quality,
                "lie": lie,
                "classification": classification,
                "distance_to_pin": pin_dist
            })
            last_lateral = lateral
            continue
        elif new_distance <= 0 and abs(new_lateral) > 40:
            result = "missed green by a mile"
            total_reward -= 6.0
            done = True
            break
        else:
            distance = new_distance
            lateral = new_lateral
            shot_num += 1
            lie = classification
        r = reward_for_shot(
            classification, distance, lateral, shot_num,
            distance, new_distance, club, mode_sample,
            hole, pin_dist, putt_attempted, last_lateral,
            shot_log, score_streak, same_club_count
        )
        total_reward += r
        total_strokes += 1
        shot_log.append({
            "shot_num": shot_num,
            "club": club,
            "aim": aim,
            "carry": carry,
            "lateral": y_offset,
            "strike": strike_quality,
            "lie": lie,
            "classification": classification,
            "distance_to_pin": pin_dist,
            "reward": r
        })
        last_lateral = lateral
        if putt_attempted:
            putts = putting_strokes(pin_dist, classification)
            total_strokes += putts
            if putts >= 3:
                total_reward -= 3.0
            total_reward -= (total_strokes - hole.par)
            result = f"green in {shot_num} ({putts} putts)"
            done = True
        if len(shot_log) >= 2:
            prev = shot_log[-2]
            if prev["club"] == club and abs(prev["carry"] - carry) < 5:
                total_reward -= 2.5
    # If not already set, set result as "other"
    if 'result' not in locals():
        if any(s.get('classification') == "Green" for s in shot_log):
            first_green = next((s['shot_num'] for s in shot_log if s['classification'] == "Green"), None)
            if first_green:
                result = f"green in {first_green}"
            else:
                result = "other"
        else:
            result = "other"
    score_delta = total_strokes - hole.par
    green_in_reg = any(s.get('classification') == "Green" for s in shot_log)
    if score_delta <= -3:
        score_label = f"albatross or better ({total_strokes} on par {hole.par})"
    elif score_delta == -2:
        score_label = f"eagle ({total_strokes} on par {hole.par})"
    elif score_delta == -1:
        score_label = f"birdie ({total_strokes} on par {hole.par})"
    elif score_delta == 0:
        score_label = f"par ({total_strokes} on par {hole.par})"
    elif score_delta == 1:
        score_label = f"bogey ({total_strokes} on par {hole.par})"
    elif score_delta == 2:
        score_label = f"double ({total_strokes} on par {hole.par})"
    elif score_delta == 3:
        score_label = f"triple ({total_strokes} on par {hole.par})"
    else:
        score_label = f"quadruple+ ({total_strokes} on par {hole.par})"
    if not green_in_reg:
        score_label = "scramble " + score_label
    return result, score_label, club_list, shot_log

def summarize_results(results, scores, club_usage):
    print("\nTop Results:")
    for res, count in Counter(results).most_common():
        print(f"{res}: {count}")
    print("\nScore Distribution:")
    for score, count in Counter(scores).most_common():
        print(f"{score}: {count}")
    print("\nClub Usage:")
    for club, count in Counter(club_usage).most_common():
        print(f"{club}: {count}")
    print("")

# =========== MAIN TRAINING LOOP ===========
def main():
    input_dim = 5 + len(clubs_all) + 1
    qnet = QNetwork(input_dim)
    optimizer = optim.Adam(qnet.parameters(), lr=0.001)
    gamma = 0.9
    epsilon_start = 0.2
    epsilon_end = 0.05
    player = Player(club_means, club_stds, lateral_stds)

    # --- Pretraining ---
    replay_buffer = load_shotlog_to_replay("shot_logs_all.csv", encode_state_action_func=encode_state_action, filter_invalid=True)
    pretrain_steps = min(20000, len(replay_buffer))
    batch_size = 64
    if pretrain_steps > 0:
        print(f"\nPretraining Q-network with {pretrain_steps} steps from replay buffer...")
        for i in range(pretrain_steps):
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
            q_values = qnet(states)
            next_q_values = qnet(next_states)
            expected_q = rewards + gamma * next_q_values * (1 - dones)
            loss = (q_values.squeeze() - expected_q.squeeze()).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 4000 == 0 or i == pretrain_steps-1:
                print(f"Pretrain step {i+1}/{pretrain_steps} (loss: {loss.item():.4f})")

    # === PHASE 1: FIXED HOLE ===
    print("\n=== PHASE 1: FIXED HOLE TRAINING (20,000 episodes) ===")
    fixed_hole = GolfHole(
        length=435,
        fairway_width=32,
        green_distance=435,
        green_depth=26,
        green_width=22,
        elevation=8,
        zones=[
            {"type":"Bunker", "x_start":240, "x_end":275, "y_start":-14, "y_end":4},
            {"type":"Tree", "x_start":120, "x_end":280, "y_start":28, "y_end":40}
        ],
        par=4,
        pin_pos=(435-13, 0)
    )

    phase1_results, phase1_scores, phase1_club_usage, phase1_shot_logs = [], [], [], []
    for ep in range(20000):
        epsilon = max(epsilon_end, epsilon_start - ep/40000*(epsilon_start-epsilon_end))
        result, score_label, club_list, shot_log = run_single_episode(
            fixed_hole, qnet, player, clubs_all, tee_clubs, epsilon, mode_list
        )
        phase1_results.append(result)
        phase1_scores.append(score_label)
        phase1_club_usage.extend(club_list)
        phase1_shot_logs.append(shot_log)
        if (ep+1) % 1000 == 0:
            print(f"Ep {ep+1} | Recent result: {result} | Score: {score_label}")

    # === PHASE 2: RANDOM HOLES ===
    print("\n=== PHASE 2: RANDOM HOLE TRAINING (20,000 episodes) ===")
    phase2_results, phase2_scores, phase2_club_usage, phase2_shot_logs = [], [], [], []
    random_hole_samples = []
    for ep in range(20000):
        hole = generate_random_hole()
        if ep < 3 or ep > 19995:
            random_hole_samples.append(hole.__dict__)
        epsilon = max(epsilon_end, epsilon_start - (ep+20000)/40000*(epsilon_start-epsilon_end))
        result, score_label, club_list, shot_log = run_single_episode(
            hole, qnet, player, clubs_all, tee_clubs, epsilon, mode_list
        )
        phase2_results.append(result)
        phase2_scores.append(score_label)
        phase2_club_usage.extend(club_list)
        phase2_shot_logs.append(shot_log)
        if (ep+1) % 1000 == 0:
            print(f"Ep {ep+1+20000} | Recent result: {result} | Score: {score_label}")

    # === SUMMARY EXPORT ===
    print("\n=== TRAINING COMPLETE ===\n")
    print("===== PHASE 1 SUMMARY: FIXED HOLE =====")
    print(f"Hole layout: {fixed_hole.__dict__}")
    summarize_results(phase1_results, phase1_scores, phase1_club_usage)
    print("===== PHASE 2 SUMMARY: RANDOM HOLES =====")
    print(f"Sample holes used: {random_hole_samples[:2]} ... {random_hole_samples[-2:]}")
    summarize_results(phase2_results, phase2_scores, phase2_club_usage)

    # Export club usage, shot logs if needed:
    with open("phase1_club_usage.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Club", "Times Used"])
        for club, count in Counter(phase1_club_usage).most_common():
            writer.writerow([club, count])
    with open("phase2_club_usage.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Club", "Times Used"])
        for club, count in Counter(phase2_club_usage).most_common():
            writer.writerow([club, count])
    with open("phase1_shot_logs.csv", "w", newline='') as csvfile:
        fieldnames = ["ep","shot_num","club","aim","carry","lateral","strike","lie","classification","distance_to_pin","reward"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ep_num, shots in enumerate(phase1_shot_logs):
            for s in shots:
                s_out = s.copy()
                s_out["ep"] = ep_num+1
                writer.writerow(s_out)
    with open("phase2_shot_logs.csv", "w", newline='') as csvfile:
        fieldnames = ["ep","shot_num","club","aim","carry","lateral","strike","lie","classification","distance_to_pin","reward"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for ep_num, shots in enumerate(phase2_shot_logs):
            for s in shots:
                s_out = s.copy()
                s_out["ep"] = ep_num+1
                writer.writerow(s_out)
    print("\nExported club usage and shot logs for both phases.")

if __name__ == "__main__":
    main()

import os

DATA_DIR = "backend/data"
os.makedirs(DATA_DIR, exist_ok=True)
