// components/AnimatedBallTrajectory.tsx
import React, { useRef, useEffect, useMemo, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

type Props = {
  start: [number, number, number];
  carry: number;
  lateral: number;
  peakHeight?: number;
  onPositionUpdate?: (pos: THREE.Vector3) => void;
};

export default function AnimatedBallTrajectory({
  start,
  carry,
  lateral,
  peakHeight = 12,
  onPositionUpdate
}: Props) {
  const ballRef = useRef<THREE.Mesh>(null);
  const [t, setT] = useState(0);
  const speed = 0.01; // Adjust for slower/faster arc

  const curve = useMemo(() => {
    const p0 = new THREE.Vector3(start[0], start[1], start[2]);
    const p1 = new THREE.Vector3(start[0] + carry / 2, start[1] + lateral / 2, start[2] + peakHeight);
    const p2 = new THREE.Vector3(start[0] + carry, start[1] + lateral, start[2]);
    return new THREE.QuadraticBezierCurve3(p0, p1, p2);
  }, [start, carry, lateral, peakHeight]);

  useFrame(() => {
    if (t < 1 && ballRef.current) {
      const pos = curve.getPoint(t);
      ballRef.current.position.set(pos.x, pos.y, pos.z);
      onPositionUpdate?.(pos);
      setT(t + speed);
    }
  });

  return (
    <>
      {/* Ball moving along the curve */}
      <mesh ref={ballRef}>
        <sphereGeometry args={[0.6, 16, 16]} />
        <meshStandardMaterial color="white" />
      </mesh>

      {/* Curve line for reference */}
      <line>
        <bufferGeometry setFromPoints={curve.getPoints(50).map(p => new THREE.Vector3(p.x, p.y, p.z))} />
        <lineBasicMaterial color="white" linewidth={2} />
      </line>
    </>
  );
}
