"""
scorecard_scraper.py
=====================

This module provides functionality to scrape golf course scorecards from
publicly available web pages. A scorecard typically lists the holes on a
course along with metadata such as par, yardage and handicap.  The
``ScorecardScraper`` class exposes a simple API for retrieving this data and
returning it in a structured format that can be consumed by the rest of the
application.

The default implementation is intentionally conservative: it looks for an
HTML table containing the word ``Hole`` in its header and then attempts to
interpret the subsequent rows as yardage and par information.  Golf course
websites vary widely, so you may need to adjust the parsing logic for
specific sources.  When scraping fails the class will raise a friendly
exception explaining what went wrong.

In addition to scraping arbitrary courses, a helper function
``scrape_popular_courses`` is provided which returns hard‑coded definitions
for a handful of well known courses.  These can be used for quick
demonstrations or unit tests when live scraping is not possible or
undesirable.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, List

import requests
from bs4 import BeautifulSoup


class ScorecardScraper:
    """Scrape scorecard information from a golf course web page.

    The primary entry point is :func:`scrape_course`, which takes a course
    URL and optionally a course name.  The method downloads the HTML,
    locates a likely scorecard table, and extracts hole level metadata.
    On success it returns a dictionary with keys ``course_name`` and
    ``holes``.  Each hole entry contains a unique ``hole_id`` (generated
    automatically), the numerical ``hole_number``, and the parsed
    ``par`` and ``yardage`` values.  If the site publishes handicap
    ratings for each hole those will be captured as ``handicap``.

    If you encounter a site that uses a radically different structure,
    consider subclassing this class and overriding ``_parse_html``.
    """

    def scrape_course(self, course_url: str, course_name: str | None = None) -> Dict:
        """Scrape a single course.

        :param course_url: Fully qualified URL of the course scorecard page.
        :param course_name: Optional override for the course name.  If not
            provided the page title will be used.
        :raises RuntimeError: If the page cannot be downloaded or no
            scorecard table can be found.
        :returns: Dictionary with keys ``course_name`` and ``holes``.
        """
        if not course_url or not course_url.strip():
            raise ValueError("course_url must be provided")

        try:
            response = requests.get(course_url, timeout=10)
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Failed to download course page: {exc}")

        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # Determine course name if not provided
        title_text = course_name or self._extract_title(soup)
        holes = self._parse_html(soup)

        if not holes:
            raise RuntimeError("Could not locate a scorecard table on the provided page.")

        return {
            "course_name": title_text,
            "holes": holes,
        }

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract a meaningful title from the page.

        A golf course site will often set the <title> tag to the name of
        the course.  If that tag is missing we fall back to a generic
        placeholder.
        """
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        return "Unnamed Course"

    def _parse_html(self, soup: BeautifulSoup) -> List[Dict]:
        """Locate and parse the scorecard table within the HTML soup.

        The parser looks for a table element containing the word
        "Hole" in its header.  It then inspects subsequent rows for
        patterns matching yardage, par and handicap.  Many websites
        arrange this data in the order: holes (top), yardage (next row),
        par (third row) and sometimes handicap (fourth row).  We try to
        accommodate minor deviations but cannot guarantee success for
        arbitrary layouts.
        """
        tables = soup.find_all("table")
        for table in tables:
            # Find all header cells in the table
            header_cells = [th.get_text(strip=True).lower() for th in table.find_all("th")]
            if not header_cells:
                continue
            if "hole" not in " ".join(header_cells):
                continue

            # Extract rows of interest
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue

            # Extract numeric values from a row of cells
            def parse_numeric_row(row):
                nums = []
                for cell in row.find_all(["td", "th"]):
                    text = cell.get_text(strip=True)
                    match = re.search(r"\d+", text)
                    if match:
                        nums.append(int(match.group()))
                return nums

            # Assume first row lists hole numbers
            hole_numbers = parse_numeric_row(rows[0])
            yardages = parse_numeric_row(rows[1]) if len(rows) > 1 else []
            pars = parse_numeric_row(rows[2]) if len(rows) > 2 else []
            handicap = parse_numeric_row(rows[3]) if len(rows) > 3 else []

            # Sanity check: expect between 9 and 18 holes
            if not (9 <= len(hole_numbers) <= 18):
                continue
            holes: List[Dict] = []
            for i, number in enumerate(hole_numbers):
                hole_data: Dict[str, int | str] = {}
                hole_data["hole_number"] = number
                # Generate a temporary hole_id; CourseManager will override later
                hole_data["hole_id"] = str(uuid.uuid4())
                # Populate yardage, par and handicap if available
                hole_data["yardage"] = yardages[i] if i < len(yardages) else None
                hole_data["par"] = pars[i] if i < len(pars) else None
                hole_data["handicap"] = handicap[i] if i < len(handicap) else None
                holes.append(hole_data)
            return holes
        return []


def scrape_popular_courses() -> List[Dict]:
    """Return a list of preconfigured popular courses.

    When live scraping is unavailable it can be useful to seed the system
    with a small set of well known courses.  The definitions below are
    intentionally simple – they include only hole number, par and yardage.
    You can extend or replace this list with real data as desired.
    """
    return [
        {
            "course_name": "Augusta National Golf Club",
            "holes": [
                {"hole_number": 1, "par": 4, "yardage": 445, "hole_id": str(uuid.uuid4())},
                {"hole_number": 2, "par": 5, "yardage": 575, "hole_id": str(uuid.uuid4())},
                {"hole_number": 3, "par": 4, "yardage": 350, "hole_id": str(uuid.uuid4())},
                {"hole_number": 4, "par": 3, "yardage": 240, "hole_id": str(uuid.uuid4())},
                {"hole_number": 5, "par": 4, "yardage": 455, "hole_id": str(uuid.uuid4())},
                {"hole_number": 6, "par": 3, "yardage": 180, "hole_id": str(uuid.uuid4())},
                {"hole_number": 7, "par": 4, "yardage": 450, "hole_id": str(uuid.uuid4())},
                {"hole_number": 8, "par": 5, "yardage": 570, "hole_id": str(uuid.uuid4())},
                {"hole_number": 9, "par": 4, "yardage": 460, "hole_id": str(uuid.uuid4())},
                {"hole_number": 10, "par": 4, "yardage": 495, "hole_id": str(uuid.uuid4())},
                {"hole_number": 11, "par": 4, "yardage": 505, "hole_id": str(uuid.uuid4())},
                {"hole_number": 12, "par": 3, "yardage": 155, "hole_id": str(uuid.uuid4())},
                {"hole_number": 13, "par": 5, "yardage": 510, "hole_id": str(uuid.uuid4())},
                {"hole_number": 14, "par": 4, "yardage": 440, "hole_id": str(uuid.uuid4())},
                {"hole_number": 15, "par": 5, "yardage": 530, "hole_id": str(uuid.uuid4())},
                {"hole_number": 16, "par": 3, "yardage": 170, "hole_id": str(uuid.uuid4())},
                {"hole_number": 17, "par": 4, "yardage": 440, "hole_id": str(uuid.uuid4())},
                {"hole_number": 18, "par": 4, "yardage": 465, "hole_id": str(uuid.uuid4())},
            ],
        },
        {
            "course_name": "St Andrews Old Course",
            "holes": [
                {"hole_number": 1, "par": 4, "yardage": 376, "hole_id": str(uuid.uuid4())},
                {"hole_number": 2, "par": 4, "yardage": 453, "hole_id": str(uuid.uuid4())},
                {"hole_number": 3, "par": 4, "yardage": 397, "hole_id": str(uuid.uuid4())},
                {"hole_number": 4, "par": 4, "yardage": 480, "hole_id": str(uuid.uuid4())},
                {"hole_number": 5, "par": 5, "yardage": 568, "hole_id": str(uuid.uuid4())},
                {"hole_number": 6, "par": 4, "yardage": 412, "hole_id": str(uuid.uuid4())},
                {"hole_number": 7, "par": 4, "yardage": 371, "hole_id": str(uuid.uuid4())},
                {"hole_number": 8, "par": 3, "yardage": 175, "hole_id": str(uuid.uuid4())},
                {"hole_number": 9, "par": 4, "yardage": 352, "hole_id": str(uuid.uuid4())},
                {"hole_number": 10, "par": 4, "yardage": 386, "hole_id": str(uuid.uuid4())},
                {"hole_number": 11, "par": 3, "yardage": 174, "hole_id": str(uuid.uuid4())},
                {"hole_number": 12, "par": 4, "yardage": 348, "hole_id": str(uuid.uuid4())},
                {"hole_number": 13, "par": 4, "yardage": 465, "hole_id": str(uuid.uuid4())},
                {"hole_number": 14, "par": 5, "yardage": 618, "hole_id": str(uuid.uuid4())},
                {"hole_number": 15, "par": 4, "yardage": 455, "hole_id": str(uuid.uuid4())},
                {"hole_number": 16, "par": 4, "yardage": 418, "hole_id": str(uuid.uuid4())},
                {"hole_number": 17, "par": 4, "yardage": 495, "hole_id": str(uuid.uuid4())},
                {"hole_number": 18, "par": 4, "yardage": 356, "hole_id": str(uuid.uuid4())},
            ],
        },
        {
            "course_name": "Pebble Beach Golf Links",
            "holes": [
                {"hole_number": 1, "par": 4, "yardage": 380, "hole_id": str(uuid.uuid4())},
                {"hole_number": 2, "par": 5, "yardage": 511, "hole_id": str(uuid.uuid4())},
                {"hole_number": 3, "par": 4, "yardage": 390, "hole_id": str(uuid.uuid4())},
                {"hole_number": 4, "par": 4, "yardage": 326, "hole_id": str(uuid.uuid4())},
                {"hole_number": 5, "par": 3, "yardage": 195, "hole_id": str(uuid.uuid4())},
                {"hole_number": 6, "par": 5, "yardage": 485, "hole_id": str(uuid.uuid4())},
                {"hole_number": 7, "par": 3, "yardage": 106, "hole_id": str(uuid.uuid4())},
                {"hole_number": 8, "par": 4, "yardage": 428, "hole_id": str(uuid.uuid4())},
                {"hole_number": 9, "par": 4, "yardage": 462, "hole_id": str(uuid.uuid4())},
                {"hole_number": 10, "par": 4, "yardage": 446, "hole_id": str(uuid.uuid4())},
                {"hole_number": 11, "par": 4, "yardage": 390, "hole_id": str(uuid.uuid4())},
                {"hole_number": 12, "par": 3, "yardage": 202, "hole_id": str(uuid.uuid4())},
                {"hole_number": 13, "par": 4, "yardage": 444, "hole_id": str(uuid.uuid4())},
                {"hole_number": 14, "par": 5, "yardage": 580, "hole_id": str(uuid.uuid4())},
                {"hole_number": 15, "par": 4, "yardage": 396, "hole_id": str(uuid.uuid4())},
                {"hole_number": 16, "par": 4, "yardage": 403, "hole_id": str(uuid.uuid4())},
                {"hole_number": 17, "par": 3, "yardage": 208, "hole_id": str(uuid.uuid4())},
                {"hole_number": 18, "par": 5, "yardage": 543, "hole_id": str(uuid.uuid4())},
            ],
        },
    ]