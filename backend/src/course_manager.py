import json
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime

class CourseManager:
    """Manages course data storage and retrieval"""
    
    def __init__(self, courses_dir: str):
        self.courses_dir = courses_dir
        os.makedirs(courses_dir, exist_ok=True)
        
    def save_course(self, course_name: str, holes_data: List[Dict]) -> str:
        """
        Save course data and return course_id
        
        Args:
            course_name: Name of the golf course
            holes_data: List of hole dictionaries with geometry data
            
        Returns:
            course_id: Unique identifier for the course
        """
        course_id = str(uuid.uuid4())
        
        # Validate and process holes data
        processed_holes = []
        for i, hole_data in enumerate(holes_data):
            hole_id = f"{course_id}_hole_{i+1}"
            
            processed_hole = self._process_hole_data(hole_data, hole_id, i+1)
            processed_holes.append(processed_hole)
        
        # Save course metadata
        course_metadata = {
            "course_id": course_id,
            "course_name": course_name,
            "created_at": datetime.now().isoformat(),
            "holes_count": len(processed_holes),
            "holes": [{"hole_id": hole["hole_id"], "hole_number": hole["hole_number"], "par": hole["par"]} for hole in processed_holes]
        }
        
        # Save files
        metadata_path = os.path.join(self.courses_dir, f"{course_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(course_metadata, f, indent=2)
        
        # Save individual hole files
        for hole in processed_holes:
            hole_path = os.path.join(self.courses_dir, f"{hole['hole_id']}.json")
            with open(hole_path, 'w') as f:
                json.dump(hole, f, indent=2)
        
        return course_id
    
    def _process_hole_data(self, hole_data: Dict, hole_id: str, hole_number: int) -> Dict:
        """Process and validate hole data"""
        
        # Required fields with defaults
        processed = {
            "hole_id": hole_id,
            "hole_number": hole_number,
            "par": hole_data.get("par", 4),
            "yardage": hole_data.get("yardage", 400),
            
            # Geometry
            "green_distance": hole_data.get("green_distance", hole_data.get("yardage", 400)),
            "green_depth": hole_data.get("green_depth", 25),
            "green_width": hole_data.get("green_width", 20),
            "fairway_width": hole_data.get("fairway_width", 30),
            "elevation": hole_data.get("elevation", 0),
            
            # Pin position (x, y) - default to center-back of green
            "pin_position": hole_data.get("pin_position", self._default_pin_position(hole_data)),
            
            # Hazards and features
            "zones": hole_data.get("zones", []),
            "ob_zones": hole_data.get("ob_zones", self._default_ob_zones(hole_data)),
            "water_zones": hole_data.get("water_zones", []),
            
            # Metadata
            "created_at": datetime.now().isoformat()
        }
        
        # Validate critical measurements
        self._validate_hole_geometry(processed)
        
        return processed
    
    def _default_pin_position(self, hole_data: Dict) -> List[float]:
        """Generate default pin position"""
        green_distance = hole_data.get("green_distance", hole_data.get("yardage", 400))
        green_depth = hole_data.get("green_depth", 25)
        
        # Default to center-back of green
        pin_x = green_distance - green_depth * 0.3  # 30% from back
        pin_y = 0  # Center of green
        
        return [pin_x, pin_y]
    
    def _default_ob_zones(self, hole_data: Dict) -> List[Dict]:
        """Generate default out-of-bounds zones"""
        green_distance = hole_data.get("green_distance", hole_data.get("yardage", 400))
        fairway_width = hole_data.get("fairway_width", 30)
        
        # Standard OB zones on left and right
        return [
            {
                "type": "OB",
                "x_start": 0,
                "x_end": green_distance + 20,
                "y_start": fairway_width/2 + 40,
                "y_end": fairway_width/2 + 100
            },
            {
                "type": "OB", 
                "x_start": 0,
                "x_end": green_distance + 20,
                "y_start": -(fairway_width/2 + 100),
                "y_end": -(fairway_width/2 + 40)
            }
        ]
    
    def _validate_hole_geometry(self, hole_data: Dict):
        """Validate hole geometry makes sense"""
        if hole_data["green_distance"] < 50:
            raise ValueError(f"Green distance too short: {hole_data['green_distance']}")
        
        if hole_data["par"] not in [3, 4, 5]:
            raise ValueError(f"Invalid par: {hole_data['par']}")
        
        if hole_data["green_width"] < 10 or hole_data["green_depth"] < 15:
            raise ValueError("Green dimensions too small")
    
    def list_courses(self) -> List[Dict]:
        """Get list of all available courses"""
        courses = []
        
        for filename in os.listdir(self.courses_dir):
            if filename.endswith("_metadata.json"):
                metadata_path = os.path.join(self.courses_dir, filename)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    courses.append(metadata)
        
        return sorted(courses, key=lambda x: x["course_name"])
    
    def get_course_holes(self, course_id: str) -> Optional[List[Dict]]:
        """Get all holes for a specific course"""
        metadata_path = os.path.join(self.courses_dir, f"{course_id}_metadata.json")
        
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        holes = []
        for hole_info in metadata["holes"]:
            hole_data = self.get_hole_data(hole_info["hole_id"])
            if hole_data:
                holes.append(hole_data)
        
        return sorted(holes, key=lambda x: x["hole_number"])
    
    def get_hole_data(self, hole_id: str) -> Optional[Dict]:
        """Get specific hole layout data"""
        hole_path = os.path.join(self.courses_dir, f"{hole_id}.json")
        
        if not os.path.exists(hole_path):
            return None
        
        with open(hole_path, 'r') as f:
            return json.load(f)
    
    def delete_course(self, course_id: str) -> bool:
        """Delete a course and all its holes"""
        try:
            # Get course metadata to find all holes
            metadata_path = os.path.join(self.courses_dir, f"{course_id}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Delete individual hole files
                for hole_info in metadata["holes"]:
                    hole_path = os.path.join(self.courses_dir, f"{hole_info['hole_id']}.json")
                    if os.path.exists(hole_path):
                        os.remove(hole_path)
                
                # Delete metadata file
                os.remove(metadata_path)
                
            return True
        except Exception:
            return False
    
    def update_hole(self, hole_id: str, updates: Dict) -> bool:
        """Update specific hole data"""
        hole_path = os.path.join(self.courses_dir, f"{hole_id}.json")
        
        if not os.path.exists(hole_path):
            return False
        
        try:
            with open(hole_path, 'r') as f:
                hole_data = json.load(f)
            
            # Update fields
            hole_data.update(updates)
            hole_data["updated_at"] = datetime.now().isoformat()
            
            # Validate updated data
            self._validate_hole_geometry(hole_data)
            
            with open(hole_path, 'w') as f:
                json.dump(hole_data, f, indent=2)
            
            return True
        except Exception:
            return False

    def import_from_scorecard_data(self, scorecard_data: Dict) -> str:
        """
        Import course from scraped scorecard data
        
        Expected format:
        {
            "course_name": "Pebble Beach Golf Links",
            "holes": [
                {
                    "hole_number": 1,
                    "par": 4,
                    "yardage": 381,
                    "description": "Slight dogleg right...",
                    "features": ["bunker_left", "water_right"]
                },
                ...
            ]
        }
        """
        
        course_name = scorecard_data["course_name"]
        raw_holes = scorecard_data["holes"]
        
        # Convert scorecard format to our hole format
        processed_holes = []
        for hole_data in raw_holes:
            processed_hole = self._convert_scorecard_hole(hole_data)
            processed_holes.append(processed_hole)
        
        return self.save_course(course_name, processed_holes)
    
    def _convert_scorecard_hole(self, scorecard_hole: Dict) -> Dict:
        """Convert scorecard hole data to our internal format"""
        
        yardage = scorecard_hole["yardage"]
        par = scorecard_hole["par"]
        hole_number = scorecard_hole["hole_number"]
        
        # Estimate geometry based on par and yardage
        if par == 3:
            fairway_width = 25
            green_size_mult = 1.2  # Par 3 greens are usually larger
        elif par == 4:
            fairway_width = 30
            green_size_mult = 1.0
        else:  # par 5
            fairway_width = 35
            green_size_mult = 0.9
        
        green_depth = int(25 * green_size_mult)
        green_width = int(22 * green_size_mult)
        
        # Generate zones based on features
        zones = self._generate_zones_from_features(
            scorecard_hole.get("features", []), 
            yardage, 
            fairway_width
        )
        
        return {
            "par": par,
            "yardage": yardage,
            "green_distance": yardage,
            "green_depth": green_depth,
            "green_width": green_width,
            "fairway_width": fairway_width,
            "elevation": 0,  # Would need elevation data from course
            "zones": zones,
            "description": scorecard_hole.get("description", ""),
            "hole_number": hole_number
        }
    
    def _generate_zones_from_features(self, features: List[str], yardage: int, fairway_width: float) -> List[Dict]:
        """Generate hazard zones based on course features"""
        zones = []
        
        for feature in features:
            if "bunker" in feature.lower():
                # Add bunker zone
                if "left" in feature:
                    zones.append({
                        "type": "Bunker",
                        "x_start": yardage * 0.4,
                        "x_end": yardage * 0.6,
                        "y_start": fairway_width/2 + 5,
                        "y_end": fairway_width/2 + 20
                    })
                elif "right" in feature:
                    zones.append({
                        "type": "Bunker", 
                        "x_start": yardage * 0.4,
                        "x_end": yardage * 0.6,
                        "y_start": -(fairway_width/2 + 20),
                        "y_end": -(fairway_width/2 + 5)
                    })
            
            elif "water" in feature.lower():
                # Add water hazard
                if "left" in feature:
                    zones.append({
                        "type": "Water",
                        "x_start": yardage * 0.2,
                        "x_end": yardage * 0.8,
                        "y_start": fairway_width/2 + 10,
                        "y_end": fairway_width/2 + 40
                    })
                elif "right" in feature:
                    zones.append({
                        "type": "Water",
                        "x_start": yardage * 0.2, 
                        "x_end": yardage * 0.8,
                        "y_start": -(fairway_width/2 + 40),
                        "y_end": -(fairway_width/2 + 10)
                    })
            
            elif "tree" in feature.lower():
                # Add tree line
                zones.append({
                    "type": "Tree",
                    "x_start": 0,
                    "x_end": yardage,
                    "y_start": fairway_width/2 + 20,
                    "y_end": fairway_width/2 + 35
                })
        
        # Always add fringe around green
        zones.append({
            "type": "Fringe",
            "x_start": yardage - 35,
            "x_end": yardage + 35,
            "y_start": -25,
            "y_end": 25
        })
        
        return zones