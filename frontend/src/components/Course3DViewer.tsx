// components/Course3DViewer.tsx
// A very simple Three.js visualisation of a golf hole.  It shows a flat
// ground plane, the pin, the current ball position, and the AI's
// recommended landing zone.  The ball can be dragged to a new position
// and the parent component is notified via the `onBallMove` callback.

import React, { useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

interface Course3DViewerProps {
  holeData: any;
  ballPosition: { x: number; y: number };
  target: any;
  onBallMove: (x: number, y: number) => void;
}

export default function Course3DViewer({ holeData, ballPosition, target, onBallMove }: Course3DViewerProps) {
  // Local state to track dragging
  const [isDragging, setDragging] = useState(false);
  const planeRef = useRef<THREE.Mesh>(null!);
  const ballRef = useRef<THREE.Mesh>(null!);

  // Convert backend shot recommendation into target coordinates.  We
  // assume that a positive expected carry moves the ball forward
  // (increasing x) and that aim_angle_degrees approximates yards left
  // or right for simplicity.
  const targetX = ballPosition.x + (target?.expected_carry || 0);
  const targetY = ballPosition.y + (target?.aim_angle_degrees || 0);

  // Event handlers for dragging the ball
  const onPointerDown = (event: any) => {
    event.stopPropagation();
    setDragging(true);
  };
  const onPointerUp = (event: any) => {
    event.stopPropagation();
    if (isDragging) {
      setDragging(false);
      // Notify parent of new ball position
      const [x, y] = event.point;
      onBallMove(x, y);
    }
  };
  const onPointerMove = (event: any) => {
    if (!isDragging) return;
    event.stopPropagation();
    // Update ball position visually during drag
    const [x, y] = event.point;
    if (ballRef.current) ballRef.current.position.set(x, y, 0.2);
  };

  return (
    <div style={{ height: 500, border: '1px solid #ccc' }}>
      <Canvas camera={{ position: [holeData.yardage ?? 150, -80, 80], fov: 45 }}>
        {/* Ground plane */}
        <mesh ref={planeRef} rotation={[-Math.PI / 2, 0, 0]} onPointerDown={onPointerDown} onPointerUp={onPointerUp} onPointerMove={onPointerMove}>
          <planeGeometry args={[holeData.yardage ? holeData.yardage * 2 : 200, 200]} />
          <meshStandardMaterial color='#228B22' />
        </mesh>
        {/* Pin flag */}
        <mesh position={[holeData.yardage ?? 100, 0, 0]}>
          <coneGeometry args={[1, 5, 8]} />
          <meshStandardMaterial color='red' />
        </mesh>
        {/* Ball */}
        <mesh
          ref={ballRef}
          position={[ballPosition.x, ballPosition.y, 0.5]}
          onPointerDown={onPointerDown}
          onPointerUp={onPointerUp}
          onPointerMove={onPointerMove}
        >
          <sphereGeometry args={[1, 32, 32]} />
          <meshStandardMaterial color='white' />
        </mesh>
        {/* AI target marker */}
        {target && (
          <mesh position={[targetX, targetY, 0.3]}>
            <sphereGeometry args={[1.5, 32, 32]} />
            <meshStandardMaterial color='green' opacity={0.5} transparent />
          </mesh>
        )}
        {/* Lighting and controls */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[50, 50, 50]} intensity={0.5} />
        <OrbitControls />
      </Canvas>
    </div>
  );
}
