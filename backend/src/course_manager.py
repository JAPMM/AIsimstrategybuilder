"""
course_manager.py
==================

The CourseManager is responsible for persisting and retrieving course data
on the server.  A course is defined by a set of holes, each with its own
identifier, number, par and yardage (and optional additional metadata such
as handicap).  Courses are stored in JSON files on disk so that they
survive server restarts.  The manager maintains an in‑memory index for
efficient lookup by course identifier or by hole identifier.

When saving a new course the manager generates a unique course id based
on the course name.  Hole identifiers are assigned by concatenating the
course id with an underscore and the hole number (e.g. ``augusta_1``).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Dict, List, Optional, Tuple


def _slugify(text: str) -> str:
    """Convert a human readable course name into a safe identifier.

    Non‑alphanumeric characters are removed and spaces are replaced with
    underscores.  If the resulting slug is empty a random UUID is used
    instead.  Slugs are always lower case.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or uuid.uuid4().hex


class CourseManager:
    """Manage the lifecycle of courses and holes.

    Courses are stored as individual JSON files in a directory provided
    on initialization.  At startup the manager scans that directory and
    loads all courses into memory.  Methods are provided to save new
    courses, list courses, fetch holes for a course and retrieve
    individual holes.
    """

    def __init__(self, courses_dir: str):
        self.courses_dir = courses_dir
        os.makedirs(self.courses_dir, exist_ok=True)
        self.courses: Dict[str, Dict] = {}
        self._hole_index: Dict[str, Dict] = {}
        self._load_courses()

    def _load_courses(self) -> None:
        """Load all course JSON files into memory.

        This method populates the ``courses`` dict and the ``_hole_index``
        mapping for fast lookup.  It is idempotent and safe to call
        multiple times.
        """
        for filename in os.listdir(self.courses_dir):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.courses_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                course_id = data.get("course_id") or filename[:-5]
                self.courses[course_id] = data
                # Index holes
                for hole in data.get("holes", []):
                    self._hole_index[hole["hole_id"]] = hole
            except Exception:
                # Skip corrupt files
                continue

    def save_course(self, course_name: str, holes: List[Dict]) -> str:
        """Persist a new course and return its identifier.

        A slug is generated from the course name and used as the base
        identifier.  If a course with the same slug already exists a
        suffix is appended to avoid collision.

        Hole identifiers are reassigned using the new course id.  The
        input hole list is not mutated – a deep copy is used.
        """
        if not course_name:
            raise ValueError("course_name must not be empty")
        base_slug = _slugify(course_name)
        course_id = base_slug
        suffix = 1
        while course_id in self.courses:
            course_id = f"{base_slug}_{suffix}"
            suffix += 1

        # Deep copy holes and assign consistent IDs
        saved_holes: List[Dict] = []
        for idx, hole in enumerate(holes, start=1):
            hole_copy = hole.copy()
            hole_copy["hole_id"] = f"{course_id}_hole_{idx}"
            hole_copy["hole_number"] = hole.get("hole_number", idx)
            saved_holes.append(hole_copy)
            self._hole_index[hole_copy["hole_id"]] = hole_copy

        course_data = {
            "course_id": course_id,
            "course_name": course_name,
            "holes": saved_holes,
        }

        # Save to disk
        path = os.path.join(self.courses_dir, f"{course_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(course_data, f, indent=2)

        self.courses[course_id] = course_data
        return course_id

    def list_courses(self) -> List[Dict]:
        """Return a summary list of all courses.

        Each entry includes the ``course_id``, ``course_name`` and
        ``holes_count``.
        """
        return [
            {
                "course_id": cid,
                "course_name": data.get("course_name"),
                "holes_count": len(data.get("holes", [])),
            }
            for cid, data in self.courses.items()
        ]

    def get_course_holes(self, course_id: str) -> List[Dict]:
        """Return the list of holes for a given course.

        If the course does not exist an empty list is returned.
        """
        course = self.courses.get(course_id)
        return course.get("holes", []) if course else []

    def get_hole_data(self, hole_id: str) -> Optional[Dict]:
        """Return the hole data for a given hole identifier.

        Holes are looked up in a precomputed index for constant time access.
        Returns ``None`` if the hole cannot be found.
        """
        return self._hole_index.get(hole_id)