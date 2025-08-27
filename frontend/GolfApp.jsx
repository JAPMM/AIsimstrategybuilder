import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Box, Plane } from '@react-three/drei';
import * as THREE from 'three';

// Golf Course 3D Visualization Component
function GolfHole({ holeData, ballPosition, onBallMove, optimalShot }) {
  const meshRef = useRef();
  const ballRef = useRef();
  const { camera, gl } = useThree();
  
  useEffect(() => {
    if (ballRef.current) {
      ballRef.current.position.set(ballPosition.x, ballPosition.y, 0.5);
    }
  }, [ballPosition]);
  
  // Handle ball dragging
  const handleBallPointerDown = (event) => {
    event.stopPropagation();
    
    const handleMouseMove = (moveEvent) => {
      const mouse = new THREE.Vector2();
      mouse.x = (moveEvent.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(moveEvent.clientY / window.innerHeight) * 2 + 1;
      
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);
      
      // Create a plane at ground level for intersection
      const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
      const intersection = new THREE.Vector3();
      raycaster.ray.intersectPlane(plane, intersection);
      
      if (intersection) {
        onBallMove({ x: intersection.x, y: intersection.y });
      }
    };
    
    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };
  
  // Render course elements
  const renderZones = () => {
    const zones = holeData.zones || [];
    
    return zones.map((zone, index) => {
      const width = zone.x_end - zone.x_start;
      const height = zone.y_end - zone.y_start;
      const centerX = (zone.x_start + zone.x_end) / 2;
      const centerY = (zone.y_start + zone.y_end) / 2;
      
      let color = '#90EE90'; // Default green
      
      switch (zone.type) {
        case 'Water':
          color = '#4169E1';
          break;
        case 'Bunker':
          color = '#F4A460';
          break;
        case 'Tree':
          color = '#228B22';
          break;
        case 'Rough':
          color = '#32CD32';
          break;
        case 'Fringe':
          color = '#7CFC00';
          break;
        case 'Green':
          color = '#00FF00';
          break;
      }
      
      return (
        <Box
          key={index}
          args={[width, height, 0.1]}
          position={[centerX, centerY, 0.05]}
        >
          <meshStandardMaterial color={color} opacity={0.7} transparent />
        </Box>
      );
    });
  };
  
  return (
    <group ref={meshRef}>
      {/* Fairway */}
      <Plane
        args={[holeData.green_distance || 400, holeData.fairway_width || 30]}
        position={[(holeData.green_distance || 400) / 2, 0, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
      >
        <meshStandardMaterial color="#90EE90" />
      </Plane>
      
      {/* Green */}
      <Box
        args={[holeData.green_depth || 25, holeData.green_width || 20, 0.1]}
        position={[
          (holeData.green_distance || 400) - (holeData.green_depth || 25) / 2,
          0,
          0.05
        ]}
      >
        <meshStandardMaterial color="#00FF00" />
      </Box>
      
      {/* Pin */}
      <Box
        args={[1, 1, 10]}
        position={[
          holeData.pin_position ? holeData.pin_position[0] : (holeData.green_distance || 400) - 12,
          holeData.pin_position ? holeData.pin_position[1] : 0,
          5
        ]}
      >
        <meshStandardMaterial color="#FF0000" />
      </Box>
      
      {/* Course zones */}
      {renderZones()}
      
      {/* Ball */}
      <mesh
        ref={ballRef}
        position={[ballPosition.x, ballPosition.y, 0.5]}
        onPointerDown={handleBallPointerDown}
        onPointerEnter={() => (gl.domElement.style.cursor = 'grab')}
        onPointerLeave={() => (gl.domElement.style.cursor = 'default')}
      >
        <sphereGeometry args={[1, 16, 16]} />
        <meshStandardMaterial color="#FFFFFF" />
      </mesh>
      
      {/* Optimal shot indicator */}
      {optimalShot && (
        <group>
          {/* Shot line */}
          <mesh>
            <cylinderGeometry args={[0.2, 0.2, optimalShot.expected_outcome?.carry_distance || 150]} />
            <meshStandardMaterial color="#FFD700" opacity={0.6} transparent />
          </mesh>
          
          {/* Target area */}
          <mesh position={[optimalShot.expected_outcome?.carry_distance || 150, optimalShot.aim_point || 0, 1]}>
            <sphereGeometry args={[3, 8, 8]} />
            <meshStandardMaterial color="#FFD700" opacity={0.4} transparent />
          </mesh>
        </group>
      )}
    </group>
  );
}

// Shot Information Panel
function ShotPanel({ ballPosition, optimalShot, shotOptions, onClubSelect }) {
  if (!optimalShot) {
    return (
      <div className="bg-white p-4 rounded-lg shadow-lg">
        <p className="text-gray-600">Loading optimal shot...</p>
      </div>
    );
  }
  
  return (
    <div className="bg-white p-4 rounded-lg shadow-lg max-w-md">
      <div className="mb-4">
        <h3 className="text-lg font-bold text-gray-800">Ball Position</h3>
        <p className="text-sm text-gray-600">
          Distance: {ballPosition.distance?.toFixed(1)} yards | 
          Lateral: {ballPosition.lateral?.toFixed(1)} yards
        </p>
        <p className="text-sm text-gray-600">Lie: {ballPosition.lie || 'Fairway'}</p>
      </div>
      
      <div className="mb-4">
        <h3 className="text-lg font-bold text-green-600">Optimal Shot</h3>
        <div className="bg-green-50 p-3 rounded">
          <p className="font-semibold">{optimalShot.club}</p>
          <p className="text-sm">Aim: {optimalShot.aim_point > 0 ? `${optimalShot.aim_point}° right` : optimalShot.aim_point < 0 ? `${Math.abs(optimalShot.aim_point)}° left` : 'straight'}</p>
          <p className="text-sm">Confidence: {optimalShot.confidence}%</p>
          <p className="text-sm">Expected carry: {optimalShot.expected_outcome?.carry_distance?.toFixed(0)} yards</p>
          {optimalShot.strokes_gained && (
            <p className="text-sm text-green-600">
              Strokes gained: {optimalShot.strokes_gained > 0 ? '+' : ''}{optimalShot.strokes_gained}
            </p>
          )}
        </div>
      </div>
      
      {shotOptions && shotOptions.length > 0 && (
        <div>
          <h4 className="font-semibold text-gray-700 mb-2">Alternative Shots</h4>
          <div className="space-y-2">
            {shotOptions.slice(0, 2).map((option, index) => (
              <div 
                key={index}
                className="bg-gray-50 p-2 rounded cursor-pointer hover:bg-gray-100 transition-colors"
                onClick={() => onClubSelect(option)}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <p className="font-medium text-sm">{option.club}</p>
                    <p className="text-xs text-gray-600">
                      Aim {option.aim_point}° | {option.expected_outcome?.carry_distance?.toFixed(0)}y
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-gray-600">{option.confidence}%</p>
                    <p className="text-xs" style={{color: option.risk_level === 'Low' ? 'green' : option.risk_level === 'High' ? 'red' : 'orange'}}>
                      {option.risk_level} Risk
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Course Selection Component
function CourseSelector({ courses, selectedCourse, onCourseSelect, onImportCourse }) {
  const [importUrl, setImportUrl] = useState('');
  const [courseName, setCourseName] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  
  const handleImport = async () => {
    if (!importUrl || !courseName) return;
    
    setIsImporting(true);
    try {
      await onImportCourse(importUrl, courseName);
      setImportUrl('');
      setCourseName('');
    } finally {
      setIsImporting(false);
    }
  };
  
  return (
    <div className="bg-white p-4 rounded-lg shadow-lg">
      <h3 className="text-lg font-bold mb-4">Select Course</h3>
      
      <div className="mb-4">
        <select
          value={selectedCourse || ''}
          onChange={(e) => onCourseSelect(e.target.value)}
          className="w-full p-2 border rounded"
        >
          <option value="">Choose a course...</option>
          {courses.map(course => (
            <option key={course.course_id} value={course.course_id}>
              {course.course_name} ({course.holes_count} holes)
            </option>
          ))}
        </select>
      </div>
      
      <div className="border-t pt-4">
        <h4 className="font-semibold mb-2">Import New Course</h4>
        <div className="space-y-2">
          <input
            type="text"
            placeholder="Course name"
            value={courseName}
            onChange={(e) => setCourseName(e.target.value)}
            className="w-full p-2 border rounded text-sm"
          />
          <input
            type="url"
            placeholder="Course website URL"
            value={importUrl}
            onChange={(e) => setImportUrl(e.target.value)}
            className="w-full p-2 border rounded text-sm"
          />
          <button
            onClick={handleImport}
            disabled={!importUrl || !courseName || isImporting}
            className="w-full bg-blue-500 text-white p-2 rounded text-sm hover:bg-blue-600 disabled:opacity-50"
          >
            {isImporting ? 'Importing...' : 'Import Course'}
          </button>
        </div>
      </div>
    </div>
  );
}

// Main Golf App Component
export default function GolfApp() {
  const [courses, setCourses] = useState([]);
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [selectedHole, setSelectedHole] = useState(null);
  const [holeData, setHoleData] = useState(null);
  const [ballPosition, setBallPosition] = useState({ x: 0, y: 0 });
  const [shotNumber, setShotNumber] = useState(1);
  const [optimalShot, setOptimalShot] = useState(null);
  const [shotOptions, setShotOptions] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  
  // Load available courses
  useEffect(() => {
    fetchCourses();
  }, []);
  
  // Load hole data when hole is selected
  useEffect(() => {
    if (selectedHole) {
      fetchHoleData(selectedHole);
    }
  }, [selectedHole]);
  
  // Reoptimize when ball position changes
  useEffect(() => {
    if (holeData && isModelLoaded) {
      reoptimizeShot();
    }
  }, [ballPosition, shotNumber, holeData, isModelLoaded]);
  
  const fetchCourses = async () => {
    try {
      const response = await fetch('http://localhost:8000/courses');
      const data = await response.json();
      setCourses(data);
    } catch (error) {
      console.error('Failed to fetch courses:', error);
    }
  };
  
  const fetchHoleData = async (holeId) => {
    try {
      const response = await fetch(`http://localhost:8000/holes/${holeId}`);
      const data = await response.json();
      setHoleData(data);
      
      // Set initial ball position (tee)
      setBallPosition({ x: 0, y: 0 });
      setShotNumber(1);
      
      // Try to load model
      await loadModel(holeId);
    } catch (error) {
      console.error('Failed to fetch hole data:', error);
    }
  };
  
  const loadModel = async (holeId) => {
    try {
      const response = await fetch(`http://localhost:8000/load-model/${holeId}`, {
        method: 'POST'
      });
      
      if (response.ok) {
        setIsModelLoaded(true);
      } else {
        // Model not found, ask user to train
        const trainResponse = confirm('No trained model found for this hole. Would you like to train it now? (This may take several minutes)');
        if (trainResponse) {
          await trainHole(holeId);
        }
      }
    } catch (error) {
      console.error('Failed to load model:', error);
    }
  };
  
  const trainHole = async (holeId) => {
    setIsTraining(true);
    try {
      const response = await fetch('http://localhost:8000/train-hole', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hole_id: holeId,
          episodes: 100000
        })
      });
      
      if (response.ok) {
        setIsModelLoaded(true);
        alert('Training completed! The AI is now ready to optimize your shots.');
      }
    } catch (error) {
      console.error('Training failed:', error);
      alert('Training failed. Please try again.');
    } finally {
      setIsTraining(false);
    }
  };
  
  const reoptimizeShot = async () => {
    if (!holeData || !isModelLoaded) return;
    
    try {
      const response = await fetch('http://localhost:8000/reoptimize-position', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hole_id: holeData.hole_id,
          ball_x: ballPosition.x,
          ball_y: ballPosition.y,
          shot_number: shotNumber
        })
      });
      
      const data = await response.json();
      setOptimalShot(data.optimal_shot);
      setShotOptions(data.alternatives || []);
      
      // Update ball position info
      setBallPosition(prev => ({
        ...prev,
        distance: data.distance_to_pin,
        lateral: data.lateral_position,
        lie: data.lie_type
      }));
      
    } catch (error) {
      console.error('Failed to reoptimize:', error);
    }
  };
  
  const handleBallMove = (newPosition) => {
    setBallPosition(newPosition);
  };
  
  const handleCourseSelect = async (courseId) => {
    setSelectedCourse(courseId);
    
    if (courseId) {
      try {
        const response = await fetch(`http://localhost:8000/courses/${courseId}/holes`);
        const holes = await response.json();
        
        if (holes.length > 0) {
          setSelectedHole(holes[0].hole_id);
        }
      } catch (error) {
        console.error('Failed to fetch course holes:', error);
      }
    }
  };
  
  const handleImportCourse = async (url, courseName) => {
    try {
      // This would integrate with the scraper
      alert('Course import feature coming soon!');
      // const response = await fetch('http://localhost:8000/scrape-course', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ url, course_name: courseName })
      // });
    } catch (error) {
      console.error('Failed to import course:', error);
    }
  };
  
  const handleClubSelect = (shotOption) => {
    // User selected a different shot option
    setOptimalShot(shotOption);
  };
  
  const handleExecuteShot = () => {
    if (!optimalShot) return;
    
    // Simulate shot execution
    const expectedCarry = optimalShot.expected_outcome?.carry_distance || 150;
    const newX = ballPosition.x + expectedCarry;
    const newY = ballPosition.y + (optimalShot.aim_point || 0);
    
    setBallPosition({ x: newX, y: newY });
    setShotNumber(prev => prev + 1);
  };
  
  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8 text-gray-800">
          AI Golf Strategy Optimizer
        </h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Course Selection */}
          <div className="lg:col-span-1">
            <CourseSelector
              courses={courses}
              selectedCourse={selectedCourse}
              onCourseSelect={handleCourseSelect}
              onImportCourse={handleImportCourse}
            />
            
            {selectedCourse && (
              <div className="mt-4 bg-white p-4 rounded-lg shadow-lg">
                <h4 className="font-semibold mb-2">Hole Selection</h4>
                <select
                  value={selectedHole || ''}
                  onChange={(e) => setSelectedHole(e.target.value)}
                  className="w-full p-2 border rounded"
                >
                  <option value="">Select hole...</option>
                  {/* This would be populated with holes from the selected course */}
                </select>
              </div>
            )}
          </div>
          
          {/* 3D Course View */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-lg p-4">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-bold">
                  {holeData ? `Hole ${holeData.hole_number} - Par ${holeData.par}` : 'Select a hole'}
                </h3>
                <div className="text-sm text-gray-600">
                  Shot #{shotNumber}
                </div>
              </div>
              
              {isTraining && (
                <div className="bg-yellow-50 border border-yellow-200 p-4 rounded mb-4">
                  <div className="flex items-center">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-yellow-600 mr-2"></div>
                    <p className="text-yellow-800">Training AI model (this may take several minutes)...</p>
                  </div>
                </div>
              )}
              
              {holeData && !isTraining && (
                <div className="h-96 border rounded">
                  <Canvas camera={{ position: [200, 100, 100], fov: 60 }}>
                    <ambientLight intensity={0.5} />
                    <directionalLight position={[100, 100, 50]} intensity={1} />
                    
                    <GolfHole
                      holeData={holeData}
                      ballPosition={ballPosition}
                      onBallMove={handleBallMove}
                      optimalShot={optimalShot}
                    />
                    
                    <OrbitControls
                      enablePan={true}
                      enableZoom={true}
                      enableRotate={true}
                      minDistance={50}
                      maxDistance={500}
                    />
                  </Canvas>
                </div>
              )}
              
              {!holeData && !isTraining && (
                <div className="h-96 border rounded flex items-center justify-center bg-gray-50">
                  <p className="text-gray-500">Select a course and hole to begin</p>
                </div>
              )}
              
              {holeData && isModelLoaded && (
                <div className="mt-4 flex justify-center space-x-4">
                  <button
                    onClick={handleExecuteShot}
                    disabled={!optimalShot}
                    className="bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600 disabled:opacity-50"
                  >
                    Execute Shot
                  </button>
                  <button
                    onClick={() => {
                      setBallPosition({ x: 0, y: 0 });
                      setShotNumber(1);
                    }}
                    className="bg-gray-500 text-white px-6 py-2 rounded hover:bg-gray-600"
                  >
                    Reset to Tee
                  </button>
                </div>
              )}
            </div>
          </div>
          
          {/* Shot Information Panel */}
          <div className="lg:col-span-1">
            <ShotPanel
              ballPosition={ballPosition}
              optimalShot={optimalShot}
              shotOptions={shotOptions}
              onClubSelect={handleClubSelect}
            />
            
            {holeData && (
              <div className="mt-4 bg-white p-4 rounded-lg shadow-lg">
                <h4 className="font-semibold mb-2">Hole Info</h4>
                <div className="text-sm space-y-1">
                  <p><span className="font-medium">Par:</span> {holeData.par}</p>
                  <p><span className="font-medium">Yardage:</span> {holeData.yardage || holeData.green_distance}</p>
                  <p><span className="font-medium">Fairway Width:</span> {holeData.fairway_width}y</p>
                  <p><span className="font-medium">Green:</span> {holeData.green_width}w × {holeData.green_depth}d</p>
                  {holeData.elevation !== 0 && (
                    <p><span className="font-medium">Elevation:</span> {holeData.elevation > 0 ? '+' : ''}{holeData.elevation}ft</p>
                  )}
                </div>
              </div>
            )}
            
            {optimalShot?.course_strategy && (
              <div className="mt-4 bg-blue-50 p-4 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Strategy</h4>
                <div className="text-sm text-blue-700 space-y-1">
                  <p><span className="font-medium">Shot Type:</span> {optimalShot.course_strategy.shot_type}</p>
                  <p><span className="font-medium">Key Focus:</span> {optimalShot.course_strategy.key_consideration}</p>
                  <p><span className="font-medium">Target:</span> {optimalShot.course_strategy.target_zone}</p>
                </div>
              </div>
            )}
            
            {!isModelLoaded && !isTraining && holeData && (
              <div className="mt-4 bg-red-50 border border-red-200 p-4 rounded-lg">
                <h4 className="font-semibold text-red-800 mb-2">AI Model Required</h4>
                <p className="text-sm text-red-700 mb-3">
                  This hole needs to be trained before the AI can provide shot recommendations.
                </p>
                <button
                  onClick={() => trainHole(holeData.hole_id)}
                  className="w-full bg-red-500 text-white p-2 rounded hover:bg-red-600"
                >
                  Train AI (100k episodes)
                </button>
                <p className="text-xs text-red-600 mt-2">
                  Training takes 5-10 minutes but only needs to be done once per hole.
                </p>
              </div>
            )}
          </div>
        </div>
        
        {/* Instructions */}
        <div className="mt-8 bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-lg font-bold mb-4">How to Use</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold text-blue-600 mb-2">1. Select Course & Hole</h4>
              <p className="text-sm text-gray-600">
                Choose a golf course and specific hole. Import new courses using the course website URL.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-blue-600 mb-2">2. Train AI (First Time)</h4>
              <p className="text-sm text-gray-600">
                Each hole needs to be trained once. The AI plays 100,000 virtual rounds to learn optimal strategy.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-blue-600 mb-2">3. Get Instant Recommendations</h4>
              <p className="text-sm text-gray-600">
                Drag the ball to any position and get instant optimal shot recommendations. Execute shots to play through the hole.
              </p>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-gray-50 rounded">
            <h4 className="font-semibold mb-2">Features</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• <strong>Drag & Drop:</strong> Move the ball anywhere on the course</li>
              <li>• <strong>Instant Reoptimization:</strong> AI instantly recalculates the best shot</li>
              <li>• <strong>Multiple Options:</strong> See alternative shot choices with risk assessment</li>
              <li>• <strong>Strokes Gained:</strong> Understand the value of each shot choice</li>
              <li>• <strong>3D Visualization:</strong> Interactive 3D course with hazards and features</li>
              <li>• <strong>Course Import:</strong> Add any course by providing its website URL</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}