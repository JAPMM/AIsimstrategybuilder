// components/Course3DViewer.tsx
import React, { useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import AnimatedBallTrajectory from "./AnimatedBallTrajectory";

// === Types ===
type CourseMesh = {
  elevation_map: number[][];
  terrain_map: number[][];
  pin_position: [number, number, number];
  wind: { speed: number; direction: number };
};

type Shot = {
  start: [number, number, number];
  carry: number;
  lateral: number;
  peakHeight?: number;
};

type StrokeZone = {
  stroke_index: number;
  zones: {
    position: [number, number]; // [x, y] in grid coords
    radius: number;
    type: "ideal" | "miss" | "avoid";
  }[];
};

type Props = {
  data: CourseMesh;
  shot?: Shot;
  strategy?: StrokeZone[];
};

// === Constants ===
const TILE_SIZE = 1;

const terrainColors = {
  0: "#3e7c3e", // rough
  1: "#4caf50", // fairway
  2: "#88e088", // green
  3: "#c2b280", // bunker
  4: "#4c84ff", // water
  5: "#999999", // OB
};

const zoneColors = {
  ideal: "#00ff00",
  miss: "#2196f3",
  avoid: "#ff1744"
};

// === Component ===
export default function Course3DViewer({ data, shot, strategy }: Props) {
  const { elevation_map, terrain_map, pin_position } = data;
  const gridX = elevation_map.length;
  const gridY = elevation_map[0].length;
  const cameraFollowRef = useRef<THREE.Vector3>(new THREE.Vector3());

  // === Camera Follow Logic ===
  const { camera } = useThree();
  const handlePositionUpdate = (pos: THREE.Vector3) => {
    cameraFollowRef.current.copy(pos);
    camera.position.lerp(new THREE.Vector3(pos.x + 10, pos.y - 30, pos.z + 15), 0.1);
    camera.lookAt(pos);
  };

  // === Terrain Geometry ===
  const terrainGeometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry();
    const vertices: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];

    for (let x = 0; x < gridX - 1; x++) {
      for (let y = 0; y < gridY - 1; y++) {
        const getColor = (terrainCode: number) =>
          new THREE.Color(terrainColors[terrainCode as keyof typeof terrainColors]);

        const points = [
          [x, y],
          [x + 1, y],
          [x, y + 1],
          [x + 1, y + 1]
        ];

        const idxBase = vertices.length / 3;

        points.forEach(([i, j]) => {
          vertices.push(i * TILE_SIZE, j * TILE_SIZE, elevation_map[i][j] * 10);
          const color = getColor(terrain_map[i][j]);
          colors.push(color.r, color.g, color.b);
        });

        indices.push(idxBase, idxBase + 1, idxBase + 2);
        indices.push(idxBase + 1, idxBase + 3, idxBase + 2);
      }
    }

    geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    return geometry;
  }, [elevation_map, terrain_map]);

  return (
    <Canvas camera={{ position: [20, -80, 80], fov: 50 }}>
      <ambientLight />
      <directionalLight position={[50, 50, 100]} intensity={0.8} />

      {/* Terrain */}
      <mesh geometry={terrainGeometry}>
        <meshStandardMaterial vertexColors={true} side={THREE.DoubleSide} />
      </mesh>

      {/* Pin */}
      <mesh position={[
        pin_position[0] * TILE_SIZE,
        pin_position[1] * TILE_SIZE,
        pin_position[2] * 10 + 1
      ]}>
        <sphereGeometry args={[0.6, 16, 16]} />
        <meshStandardMaterial color="red" />
      </mesh>

      {/* Strategy Zones */}
      {strategy?.map(({ stroke_index, zones }) =>
        zones.map((zone, i) => (
          <mesh
            key={`zone-${stroke_index}-${i}`}
            position={[zone.position[0], zone.position[1], 0.5]}
          >
            <cylinderGeometry args={[zone.radius, zone.radius, 0.2, 32]} />
            <meshStandardMaterial
              color={zoneColors[zone.type]}
              transparent
              opacity={zone.type === "avoid" ? 0.3 : 0.5}
            />
          </mesh>
        ))
      )}

      {/* Animated Shot */}
      {shot && (
        <AnimatedBallTrajectory
          start={shot.start}
          carry={shot.carry}
          lateral={shot.lateral}
          peakHeight={shot.peakHeight || 12}
          onPositionUpdate={handlePositionUpdate}
        />
      )}

      <OrbitControls />
    </Canvas>
  );
}
