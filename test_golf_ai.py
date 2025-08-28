#!/usr/bin/env python3
"""
Golf AI System Test Suite
Tests all major functionality
"""

import json
import time
import requests
import sys

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_course_import():
    """Test course import"""
    print("🔍 Testing course import...")
    
    course_data = {
        "course_name": "Test Course",
        "holes": [
            {
                "hole_number": 1,
                "par": 4,
                "yardage": 400,
                "green_distance": 400,
                "green_depth": 25,
                "green_width": 20,
                "fairway_width": 30,
                "elevation": 0,
                "pin_position": [390, 0],
                "zones": [
                    {
                        "type": "Bunker",
                        "x_start": 180,
                        "x_end": 210,
                        "y_start": -15,
                        "y_end": 0
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/courses/import", json=course_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Course import passed: {data['course_id']}")
            return data['course_id']
        else:
            print(f"❌ Course import failed: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Course import failed: {e}")
        return None

def test_popular_courses_import():
    """Test popular courses import"""
    print("🔍 Testing popular courses import...")
    
    try:
        response = requests.post(f"{BASE_URL}/import-popular-courses", timeout=15)
        if response.status_code == 200:
            data = response.json()
            courses = data.get('courses', [])
            print(f"✅ Popular courses import passed: {len(courses)} courses")
            return courses[0]['course_id'] if courses else None
        else:
            print(f"❌ Popular courses import failed: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Popular courses import failed: {e}")
        return None

def test_course_listing(course_id):
    """Test course listing"""
    print("🔍 Testing course listing...")
    
    try:
        response = requests.get(f"{BASE_URL}/courses", timeout=5)
        if response.status_code == 200:
            courses = response.json()
            found_course = any(c['course_id'] == course_id for c in courses)
            if found_course:
                print(f"✅ Course listing passed: Found {len(courses)} courses")
                return True
            else:
                print(f"❌ Course listing failed: Course {course_id} not found")
                return False
        else:
            print(f"❌ Course listing failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Course listing failed: {e}")
        return False

def test_hole_data(course_id):
    """Test hole data retrieval"""
    print("🔍 Testing hole data retrieval...")
    
    try:
        # Get course holes
        response = requests.get(f"{BASE_URL}/courses/{course_id}/holes", timeout=5)
        if response.status_code == 200:
            holes = response.json()
            if holes:
                hole_id = holes[0]['hole_id']
                print(f"✅ Got course holes: {len(holes)} holes")
                
                # Get specific hole data
                hole_response = requests.get(f"{BASE_URL}/holes/{hole_id}", timeout=5)
                if hole_response.status_code == 200:
                    hole_data = hole_response.json()
                    print(f"✅ Hole data retrieval passed: Hole {hole_data.get('hole_number', '?')} Par {hole_data.get('par', '?')}")
                    return hole_id
                else:
                    print(f"❌ Hole data retrieval failed: {hole_response.status_code}")
                    return None
            else:
                print("❌ No holes found in course")
                return None
        else:
            print(f"❌ Course holes retrieval failed: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Hole data retrieval failed: {e}")
        return None

def test_shot_optimization(hole_id):
    """Test shot optimization"""
    print("🔍 Testing shot optimization...")
    
    shot_request = {
        "hole_id": hole_id,
        "distance_to_pin": 150,
        "lateral_position": 5,
        "shot_number": 2,
        "lie_type": "Fairway",
        "player_mode": "Normal"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/optimize-shot", json=shot_request, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Shot optimization passed:")
            print(f"   Club: {data.get('club', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A')}%")
            print(f"   Aim: {data.get('aim_point', 'N/A')}°")
            print(f"   Expected carry: {data.get('expected_outcome', {}).get('carry_distance', 'N/A')} yards")
            return True
        else:
            print(f"❌ Shot optimization failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Shot optimization failed: {e}")
        return False

def test_shot_options(hole_id):
    """Test shot options"""
    print("🔍 Testing shot options...")
    
    try:
        response = requests.get(
            f"{BASE_URL}/shot-options/{hole_id}?distance=120&lateral=0&shot_num=2&top_n=3", 
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            options = data.get('options', [])
            print(f"✅ Shot options passed: {len(options)} options returned")
            for i, option in enumerate(options[:2], 1):
                print(f"   Option {i}: {option.get('club', 'N/A')} ({option.get('confidence', 'N/A')}% confidence)")
            return True
        else:
            print(f"❌ Shot options failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Shot options failed: {e}")
        return False

def test_position_reoptimization(hole_id):
    """Test position reoptimization"""
    print("🔍 Testing position reoptimization...")
    
    position_request = {
        "hole_id": hole_id,
        "ball_x": 200,
        "ball_y": 10,
        "shot_number": 2
    }
    
    try:
        response = requests.post(f"{BASE_URL}/reoptimize-position", json=position_request, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Position reoptimization passed:")
            print(f"   Distance to pin: {data.get('distance_to_pin', 'N/A')} yards")
            print(f"   Lie: {data.get('lie_type', 'N/A')}")
            optimal = data.get('optimal_shot', {})
            print(f"   Recommended: {optimal.get('club', 'N/A')} ({optimal.get('confidence', 'N/A')}%)")
            return True
        else:
            print(f"❌ Position reoptimization failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Position reoptimization failed: {e}")
        return False

def test_model_training(hole_id):
    """Test model training"""
    print("🔍 Testing model training...")
    
    training_request = {
        "hole_id": hole_id,
        "episodes": 1000  # Reduced for testing
    }
    
    try:
        response = requests.post(f"{BASE_URL}/train-hole", json=training_request, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model training passed:")
            print(f"   Episodes: {data.get('episodes', 'N/A')}")
            stats = data.get('training_stats', {})
            print(f"   Training time: {stats.get('training_time', 'N/A')}s")
            print(f"   Performance: {stats.get('score_vs_par', 'N/A')} vs par")
            return True
        else:
            print(f"❌ Model training failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Model training failed: {e}")
        return False

def run_full_test_suite():
    """Run complete test suite"""
    print("🧪 Golf AI System Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health()))
    if not results[-1][1]:
        print("❌ Backend is not running. Please start it first with:")
        print("   python start_golf_ai.py")
        return False
    
    # Test 2: Popular courses import
    course_id = test_popular_courses_import()
    results.append(("Popular Courses Import", course_id is not None))
    
    if not course_id:
        # Fallback: Try manual course import
        course_id = test_course_import()
        results.append(("Manual Course Import", course_id is not None))
    
    if not course_id:
        print("❌ Could not import any courses")
        return False
    
    # Test 3: Course listing
    results.append(("Course Listing", test_course_listing(course_id)))
    
    # Test 4: Hole data retrieval
    hole_id = test_hole_data(course_id)
    results.append(("Hole Data Retrieval", hole_id is not None))
    
    if not hole_id:
        print("❌ Could not retrieve hole data")
        return False
    
    # Test 5: Shot optimization
    results.append(("Shot Optimization", test_shot_optimization(hole_id)))
    
    # Test 6: Shot options
    results.append(("Shot Options", test_shot_options(hole_id)))
    
    # Test 7: Position reoptimization
    results.append(("Position Reoptimization", test_position_reoptimization(hole_id)))
    
    # Test 8: Model training (optional)
    train_test = input("\n❓ Test model training? (takes ~30s) [y/N]: ").lower().strip()
    if train_test == 'y':
        results.append(("Model Training", test_model_training(hole_id)))
    
    # Results summary
    print("\n" + "=" * 50)
    print("🏁 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, passed_test in results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status} - {test_name}")
        if passed_test:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n🚀 System is fully operational!")
        print("\n📚 Next Steps:")
        print("   • Access API docs: http://localhost:8000/docs")
        print("   • Import your own courses via API")
        print("   • Train models for better accuracy")
        print("   • Build your frontend application")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("   Check the error messages above for details")
        return False

def main():
    """Main test routine"""
    success = run_full_test_suite()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()