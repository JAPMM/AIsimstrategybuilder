import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";

// Lazy-load Three.js canvas component to avoid SSR issues
const Course3DViewer = dynamic(() => import("../components/Course3DViewer"), { ssr: false });

export default function HoleViewer() {
  const [strategy, setStrategy] = useState(null);
  const [courseData, setCourseData] = useState(null);

  const [selectedHole, setSelectedHole] = useState(1);

useEffect(() => {
  fetch(`/strategies/hole_${selectedHole}_strategy.json`)
    .then((res) => res.json())
    .then((data) => setStrategy(data.stroke_zones));

  fetch("/sampleCourse.json")
    .then((res) => res.json())
    .then((data) => setCourseData(data));
}, [selectedHole]);
  <select onChange={(e) => setSelectedHole(Number(e.target.value))} className="p-2 mb-4">
    {[...Array(18)].map((_, i) => (
      <option key={i} value={i + 1}>
        Hole {i + 1}
      </option>
    ))}
  </select>




  if (!strategy || !courseData) return <p>Loading viewer...</p>;

  return (
    <div className="h-screen w-screen">
      <Course3DViewer data={courseData} strategy={strategy} />
    </div>
  );
}
