import torch
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from backend.your_previous_model import (
    QNetwork, Player, GolfHole, run_single_episode, 
    club_means, club_stds, lateral_stds, clubs_all, mode_list,
    load_shotlog_to_replay, encode_state_action
)

def train_model_for_hole(hole_data: Dict, 
                        shot_data: List[Dict] = None, 
                        episodes: int = 100000) -> Tuple[QNetwork, Dict]:
    """
    Train AI model for a specific hole
    
    Args:
        hole_data: Hole geometry and layout data
        shot_data: Optional TrackMan shot data for pretraining
        episodes: Number of training episodes (default 100k)
    
    Returns:
        (trained_qnet, training_stats)
    """
    
    print(f"Training model for hole {hole_data['hole_id']} ({episodes} episodes)")
    
    # Initialize Q-network
    input_dim = 5 + len(clubs_all) + 1
    qnet = QNetwork(input_dim)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=0.001)
    gamma = 0.9
    
    # Initialize player
    player = Player(club_means, club_stds, lateral_stds)
    
    # Convert hole_data to GolfHole object
    golf_hole = create_golf_hole_from_data(hole_data)
    
    # Pretraining with shot data if available
    pretrain_steps = 0
    if shot_data:
        pretrain_steps = pretrain_with_shot_data(qnet, optimizer, shot_data, gamma)
    
    # Training phases
    training_stats = {
        "pretrain_steps": pretrain_steps,
        "total_episodes": episodes,
        "results": [],
        "scores": [],
        "final_performance": {}
    }
    
    # Main training loop
    epsilon_start = 0.3
    epsilon_end = 0.05
    
    results = []
    scores = []
    
    print("Starting main training loop...")
    for episode in range(episodes):
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_start - episode/(episodes*0.8) * (epsilon_start - epsilon_end))
        
        # Run episode
        result, score_label, club_list, shot_log = run_single_episode(
            golf_hole, qnet, player, clubs_all, 
            [c for c in clubs_all if club_means[c] >= 180],  # tee_clubs
            epsilon, mode_list
        )
        
        results.append(result)
        scores.append(score_label)
        
        # Training step (simplified - you might want to add replay buffer)
        if len(shot_log) > 1:
            train_step_from_episode(qnet, optimizer, shot_log, gamma, hole_data)
        
        # Progress logging
        if (episode + 1) % 5000 == 0:
            recent_results = results[-1000:]
            print(f"Episode {episode + 1}/{episodes} | Epsilon: {epsilon:.3f} | Recent: {result}")
            print(f"  Last 1000 episodes - Green in reg: {sum(1 for r in recent_results if 'green in' in r)}")
    
    # Final performance analysis
    final_1000 = results[-1000:]
    performance_metrics = analyze_performance(final_1000, scores[-1000:])
    
    training_stats["results"] = results
    training_stats["scores"] = scores  
    training_stats["final_performance"] = performance_metrics
    
    print(f"\nTraining completed!")
    print(f"Final 1000 episodes performance:")
    for metric, value in performance_metrics.items():
        print(f"  {metric}: {value}")
    
    return qnet, training_stats

def create_golf_hole_from_data(hole_data: Dict) -> GolfHole:
    """Convert hole_data dict to GolfHole object"""
    
    return GolfHole(
        length=hole_data.get("yardage", 400),
        fairway_width=hole_data.get("fairway_width", 30),
        green_distance=hole_data.get("green_distance", hole_data.get("yardage", 400)),
        green_depth=hole_data.get("green_depth", 25),
        green_width=hole_data.get("green_width", 20),
        elevation=hole_data.get("elevation", 0),
        zones=hole_data.get("zones", []),
        par=hole_data.get("par", 4),
        pin_pos=tuple(hole_data.get("pin_position", [hole_data.get("green_distance", 400) - 12, 0])),
        ob_zones=hole_data.get("ob_zones", []),
        water_zones=hole_data.get("water_zones", [])
    )

def pretrain_with_shot_data(qnet: QNetwork, 
                           optimizer: torch.optim.Optimizer, 
                           shot_data: List[Dict], 
                           gamma: float) -> int:
    """Pretrain network with TrackMan shot data"""
    
    if not shot_data:
        return 0
    
    print("Pretraining with TrackMan shot data...")
    
    # Convert shot data to DataFrame for easier processing
    df = pd.DataFrame(shot_data)
    
    # Save to CSV format for compatibility with existing replay loader
    temp_csv = "temp_shot_data.csv"
    df.to_csv(temp_csv, index=False)
    
    try:
        # Load into replay buffer using existing function
        replay_buffer = load_shotlog_to_replay(temp_csv, encode_state_action)
        
        # Pretrain steps
        pretrain_steps = min(10000, len(replay_buffer))
        batch_size = 64
        
        if pretrain_steps > 0:
            print(f"Pretraining with {pretrain_steps} steps from shot data...")
            
            for step in range(pretrain_steps):
                if len(replay_buffer) < batch_size:
                    break
                    
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.tensor(states, dtype=torch.float32)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)
                
                # Forward pass
                q_values = qnet(states)
                next_q_values = qnet(next_states)
                expected_q = rewards + gamma * next_q_values * (1 - dones)
                
                # Loss and backprop
                loss = (q_values.squeeze() - expected_q.squeeze()).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (step + 1) % 2000 == 0:
                    print(f"  Pretrain step {step + 1}/{pretrain_steps} (loss: {loss.item():.4f})")
        
        return pretrain_steps
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

def train_step_from_episode(qnet: QNetwork, 
                           optimizer: torch.optim.Optimizer,
                           shot_log: List[Dict], 
                           gamma: float,
                           hole_data: Dict):
    """Perform training step from episode data"""
    
    if len(shot_log) < 2:
        return
    
    # Convert shot log to training examples
    states = []
    targets = []
    
    for i in range(len(shot_log) - 1):
        shot = shot_log[i]
        next_shot = shot_log[i + 1]
        
        # Encode state-action
        state = encode_state_action(
            0,  # hole_idx
            shot.get("distance_to_pin", 0),
            shot.get("lateral", 0), 
            shot.get("shot_num", 1),
            2,  # mode encoding (Normal)
            shot.get("club", "7 Iron"),
            shot.get("aim", 0)
        )
        
        # Target is immediate reward + discounted next value
        immediate_reward = shot.get("reward", 0)
        
        # Estimate next state value (simplified)
        next_value = immediate_reward  # Could use next Q-value here
        target = immediate_reward + gamma * next_value
        
        states.append(state)
        targets.append(target)
    
    if states:
        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        # Forward pass
        predictions = qnet(states_tensor).squeeze()
        
        # Loss and backprop
        loss = torch.nn.functional.mse_loss(predictions, targets_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def analyze_performance(results: List[str], scores: List[str]) -> Dict:
    """Analyze training performance"""
    
    total_episodes = len(results)
    
    # Count different result types
    green_in_reg = sum(1 for r in results if "green in" in r.lower())
    pars = sum(1 for s in scores if "par" in s.lower())
    birdies = sum(1 for s in scores if "birdie" in s.lower())
    bogeys = sum(1 for s in scores if "bogey" in s.lower())
    
    return {
        "total_episodes": total_episodes,
        "green_in_regulation_rate": round(green_in_reg / total_episodes * 100, 1),
        "par_rate": round(pars / total_episodes * 100, 1), 
        "birdie_rate": round(birdies / total_episodes * 100, 1),
        "bogey_or_worse_rate": round(bogeys / total_episodes * 100, 1),
        "most_common_result": max(set(results), key=results.count),
        "most_common_score": max(set(scores), key=scores.count)
    }

def save_model(qnet: QNetwork, hole_id: str, models_dir: str) -> str:
    """Save trained model"""
    model_path = os.path.join(models_dir, f"{hole_id}.pth")
    torch.save({
        'model_state_dict': qnet.state_dict(),
        'hole_id': hole_id,
        'input_dim': 5 + len(clubs_all) + 1,
        'saved_at': pd.Timestamp.now().isoformat()
    }, model_path)
    
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path: str) -> QNetwork:
    """Load trained model"""
    checkpoint = torch.load(model_path)
    
    input_dim = checkpoint.get('input_dim', 5 + len(clubs_all) + 1)
    qnet = QNetwork(input_dim)
    qnet.load_state_dict(checkpoint['model_state_dict'])
    qnet.eval()
    
    print(f"Model loaded from {model_path}")
    return qnet

# Backwards compatibility with existing train_model function
def train_model(shot_data=None):
    """Legacy function for backwards compatibility"""
    if shot_data:
        df = pd.DataFrame(shot_data)
        df.to_csv("backend/shot_logs_all.csv", index=False)

    # Import and run full training loop
    from backend.your_previous_model import main as full_training_loop
    full_training_loop()
    return {"status": "Training complete (legacy mode)"}

def export_strategy_to_json(hole_id: str, qnet: QNetwork, hole_data: Dict):
    """Export trained strategy as JSON for frontend"""
from src.instant_optimizer import InstantGolfOptimizer
    
    optimizer = InstantGolfOptimizer(qnet, hole_data)
    
    # Generate strategy for different positions
    strategy_zones = []
    
    # Sample positions across the hole
    green_distance = hole_data.get("green_distance", 400)
    fairway_width = hole_data.get("fairway_width", 30)
    
    positions = [
        (green_distance, 0),  # Tee shot
        (green_distance * 0.7, 0),  # Layup position
        (green_distance * 0.4, 0),  # Approach position  
        (100, 0),  # Short iron
        (50, 0),   # Wedge
    ]
    
    for distance, lateral in positions:
        if distance > 0:
            optimal = optimizer.get_optimal_shot(distance, lateral)
            strategy_zones.append({
                "position": {"distance": distance, "lateral": lateral},
                "optimal_shot": optimal
            })
    
    # Export to frontend
    export_path = f"../frontend/public/strategies/{hole_id}_strategy.json"
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    strategy_data = {
        "hole_id": hole_id,
        "generated_at": pd.Timestamp.now().isoformat(),
        "strategy_zones": strategy_zones
    }
    
    with open(export_path, "w") as f:
        import json
        json.dump(strategy_data, f, indent=2)
    
    print(f"Strategy exported to {export_path}")