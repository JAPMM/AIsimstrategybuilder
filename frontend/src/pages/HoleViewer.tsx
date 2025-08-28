// pages/HoleViewer.tsx
// Displays a single hole using the Course3DViewer component.  It loads
// hole data from the backend, requests an optimal shot and updates it
// when the ball is moved.  Designed to be embedded in other pages.

import React, { useEffect, useState } from 'react';
import { getHoleData, optimizeShot, reoptimizeShot } from '../lib/api';
import Course3DViewer from '../components/Course3DViewer';

interface HoleViewerProps {
  holeId: string;
}

interface HoleData {
  hole_id: string;
  hole_number: number;
  par?: number;
  yardage?: number;
  pin_position?: [number, number, number];
}

export default function HoleViewer({ holeId }: HoleViewerProps) {
  const [holeData, setHoleData] = useState<HoleData | null>(null);
  const [target, setTarget] = useState<any | null>(null);
  const [ballPosition, setBallPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [loading, setLoading] = useState<boolean>(true);

  // Fetch hole data when hole changes
  useEffect(() => {
    setLoading(true);
    getHoleData(holeId)
      .then((data) => {
        setHoleData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  }, [holeId]);

  // Request the first shot when hole data is ready
  useEffect(() => {
    if (!holeData) return;
    const dist = holeData.yardage ?? 0;
    optimizeShot({
      hole_id: holeId,
      distance_to_pin: dist,
      lateral_position: 0,
      shot_number: 1,
      lie_type: 'Fairway',
      player_mode: 'Normal',
    })
      .then((res) => setTarget(res))
      .catch((err) => console.error(err));
  }, [holeData]);

  // Handle ball movement and reoptimise
  const handleBallMove = (x: number, y: number) => {
    setBallPosition({ x, y });
    reoptimizeShot({ hole_id: holeId, ball_x: x, ball_y: y, shot_number: 1 })
      .then((res) => setTarget(res))
      .catch((err) => console.error(err));
  };

  if (loading || !holeData || !target) return <p>Loading hole...</p>;

  return (
    <Course3DViewer
      holeData={holeData}
      ballPosition={ballPosition}
      target={target}
      onBallMove={handleBallMove}
    />
  );
}
