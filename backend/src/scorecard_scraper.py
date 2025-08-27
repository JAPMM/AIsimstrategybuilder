import requests
from bs4 import BeautifulSoup
import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

@dataclass
class HoleInfo:
    hole_number: int
    par: int
    yardage: int
    description: str = ""
    features: List[str] = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = []

class ScorecardScraper:
    """Scraper for golf course scorecard data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_course(self, course_url: str, course_name: str = None) -> Dict:
        """
        Scrape course data from URL
        
        Args:
            course_url: URL to course website or scorecard page
            course_name: Optional course name override
            
        Returns:
            Dictionary with course data compatible with CourseManager
        """
        
        try:
            print(f"Scraping course data from: {course_url}")
            response = self.session.get(course_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Detect course type and use appropriate scraping method
            if "pebblebeach.com" in course_url:
                return self._scrape_pebble_beach(soup, course_name)
            elif "pinehurst.com" in course_url:
                return self._scrape_pinehurst(soup, course_name)
            elif "augustanational" in course_url.lower():
                return self._scrape_augusta(soup, course_name)
            else:
                # Generic scraping method
                return self._scrape_generic(soup, course_url, course_name)
                
        except Exception as e:
            print(f"Error scraping {course_url}: {e}")
            raise

    def _scrape_pebble_beach(self, soup: BeautifulSoup, course_name: str = None) -> Dict:
        """Scrape Pebble Beach Golf Links"""
        
        course_name = course_name or "Pebble Beach Golf Links"
        holes = []
        
        # Look for hole information sections
        hole_sections = soup.find_all(['div', 'section'], class_=re.compile(r'hole', re.I))
        
        for i, section in enumerate(hole_sections[:18]):  # Max 18 holes
            hole_num = i + 1
            
            # Extract yardage
            yardage_text = section.get_text()
            yardage_match = re.search(r'(\d{3,4})\s*yard', yardage_text, re.I)
            yardage = int(yardage_match.group(1)) if yardage_match else self._estimate_yardage(hole_num)
            
            # Extract par
            par_match = re.search(r'par\s*(\d)', yardage_text, re.I)
            par = int(par_match.group(1)) if par_match else self._estimate_par(yardage)
            
            # Extract features
            features = self._extract_features(yardage_text)
            
            # Get description
            description = self._clean_description(section.get_text()[:200])
            
            holes.append({
                "hole_number": hole_num,
                "par": par,
                "yardage": yardage,
                "description": description,
                "features": features
            })
        
        return {
            "course_name": course_name,
            "holes": holes
        }

    def _scrape_generic(self, soup: BeautifulSoup, course_url: str, course_name: str = None) -> Dict:
        """Generic scraping for unknown course websites"""
        
        # Try to extract course name from page
        if not course_name:
            title_tag = soup.find('title')
            h1_tag = soup.find('h1')
            
            if title_tag:
                course_name = title_tag.get_text().strip()
            elif h1_tag:
                course_name = h1_tag.get_text().strip()
            else:
                course_name = "Unknown Course"
        
        holes = []
        
        # Look for scorecard tables
        tables = soup.find_all('table')
        scorecard_table = None
        
        for table in tables:
            table_text = table.get_text().lower()
            if any(word in table_text for word in ['hole', 'par', 'yards', 'yardage']):
                scorecard_table = table
                break
        
        if scorecard_table:
            holes = self._parse_scorecard_table(scorecard_table)
        
        # If no table found, try to find hole information in divs/sections
        if not holes:
            holes = self._extract_holes_from_content(soup)
        
        # If still no holes, create default 18-hole course
        if not holes:
            print("No hole data found, creating default course layout")
            holes = self._create_default_holes()
        
        return {
            "course_name": course_name,
            "holes": holes
        }

    def _parse_scorecard_table(self, table) -> List[Dict]:
        """Parse scorecard data from HTML table"""
        holes = []
        
        rows = table.find_all('tr')
        if len(rows) < 2:
            return holes
        
        # Find header row to identify columns
        header_row = rows[0]
        headers = [th.get_text().strip().lower() for th in header_row.find_all(['th', 'td'])]
        
        # Find column indices
        hole_col = self._find_column_index(headers, ['hole', 'no', '#'])
        par_col = self._find_column_index(headers, ['par'])
        yardage_