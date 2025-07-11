import aiohttp
import asyncio
from bs4 import BeautifulSoup
import re
from typing import Dict, Any, Optional
from urllib.parse import urljoin
from .source import DataSource

class StelleDoppieDataSource(DataSource):
    """Data source implementation using Stelle Doppie web scraping."""
    
    def __init__(self, session: aiohttp.ClientSession):
        """
        Initialize the web data source.
        
        Args:
            session: aiohttp ClientSession for making HTTP requests
        """
        self.session = session
        self.base_url = "https://www.stelledoppie.it/"
    
    def _clean_wds_name(self, wds_name: str) -> Optional[str]:
        """Clean WDS name for URL usage."""
        if not wds_name or wds_name == '':
            return None
        
        # Replace spaces with + for URL
        wds_clean = str(wds_name).strip().replace(' ', '+')
        return wds_clean
    
    async def _fetch_star_page(self, wds_name: str, max_retries: int = 3) -> Optional[str]:
        """Fetch the HTML content for a star's detail page."""
        wds_q = self._clean_wds_name(wds_name)
        if not wds_q:
            return None
        
        search_url = f"{self.base_url}index2.php?cerca_database={wds_q}&azione=cerca_testo_database&nofilter=1&section=2&ricerca=+Search+"
        
        for attempt in range(max_retries):
            try:
                async with self.session.get(search_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    response.raise_for_status()
                    content = await response.text()
                    
                    final_url = str(response.url)
                    
                    # Check if redirected to detail page
                    if 'iddoppia=' in final_url:
                        # Fix malformed URLs
                        if 'index2.php&iddoppia=' in final_url:
                            final_url = final_url.replace('index2.php&iddoppia=', 'index2.php?iddoppia=')
                        
                        # Fetch corrected URL if needed
                        if final_url != str(response.url):
                            async with self.session.get(final_url, timeout=aiohttp.ClientTimeout(total=10)) as corrected_response:
                                corrected_response.raise_for_status()
                                content = await corrected_response.text()
                        
                        return content
                    
                    # Search for detail page link
                    soup = BeautifulSoup(content, 'html.parser')
                    links = soup.find_all('a', href=True)
                    
                    for link in links:
                        href = link['href']
                        if 'iddoppia=' in href and 'report' not in href and 'section=4' not in href:
                            full_url = urljoin(self.base_url, href)
                            
                            async with self.session.get(full_url, timeout=aiohttp.ClientTimeout(total=10)) as detail_response:
                                detail_response.raise_for_status()
                                detail_content = await detail_response.text()
                                return detail_content
                    
                    return None
                    
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
            except Exception:
                return None
    
    def _extract_star_data(self, html_content: str, wds_name: str) -> Dict[str, Any]:
        """Extract star data from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        data = {
            'wds_name': wds_name,
            'date_first': None,
            'pa_first': None,
            'sep_first': None,
            'date_last': None,
            'pa_last': None,
            'sep_last': None,
            'obs': None
        }
        
        try:
            text = soup.get_text()
            
            # Patterns for extracting data
            patterns = [
                (r'Date first\s*:?\s*(\d+)', 'date_first'),
                (r'Pa first\s*:?\s*([\d.-]+)', 'pa_first'),
                (r'Sep first\s*:?\s*([\d.-]+)', 'sep_first'),
                (r'Date last\s*:?\s*(\d+)', 'date_last'),
                (r'Pa last\s*:?\s*([\d.-]+)', 'pa_last'),
                (r'Sep last\s*:?\s*([\d.-]+)', 'sep_last'),
                (r'Obs\s*:?\s*(\d+)', 'obs'),
            ]
            
            # Apply patterns
            for pattern, field in patterns:
                if data[field] is None:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            if field in ['date_first', 'date_last', 'obs']:
                                data[field] = int(match.group(1))
                            else:
                                data[field] = float(match.group(1))
                        except ValueError:
                            continue
            
            # Also check tables
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        cell_text = cells[0].get_text().strip().lower()
                        value_text = cells[1].get_text().strip()
                        
                        field_map = {
                            'date first': 'date_first',
                            'pa first': 'pa_first',
                            'sep first': 'sep_first',
                            'date last': 'date_last',
                            'pa last': 'pa_last',
                            'sep last': 'sep_last',
                            'obs': 'obs'
                        }
                        
                        for key, field in field_map.items():
                            if key in cell_text and data[field] is None:
                                try:
                                    if field in ['date_first', 'date_last', 'obs']:
                                        data[field] = int(re.search(r'(\d+)', value_text).group(1))
                                    else:
                                        data[field] = float(re.search(r'([\d.-]+)', value_text).group(1))
                                except:
                                    pass
            
            return data
            
        except Exception:
            return data
    
    def _extract_orbital_elements(self, html_content: str) -> Dict[str, Any]:
        """Extract orbital elements from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        orbital_elements = {
            'P': None,      # Period
            'T': None,      # Time of periastron
            'e': None,      # Eccentricity
            'a': None,      # Semi-major axis
            'i': None,      # Inclination
            'Omega': None,  # Longitude of ascending node
            'omega': None   # Argument of periastron
        }
        
        try:
            # Get all text content from the body, or a specific section if available
            # (e.g., if elements are inside a specific div or table-like structure)
            # For this format, getting all text content is often sufficient if patterns are unique.
            text = soup.get_text()
            
            # Define refined patterns for orbital elements based on the image format.
            # Using non-greedy matches (.*?) and capturing groups.
            # Need to be careful with spaces, newlines, and unit characters (y, a, °)
            patterns = [
                # Period (P) - Look for 'Period (P)' followed by a number and 'y'
                (r'Period \(P\)\s*([\d.]+)\s*y', 'P'),
                # Periastron (T) - Look for 'Periastron (T)' followed by a number and 'y'
                (r'Periastron \(T\)\s*([\d.]+)\s*y', 'T'),
                # Semi-major axis (a) - Look for 'Semi-major axis \(a\)' followed by a number and 'a'
                (r'Semi-major axis \(a\)\s*([\d.]+)\s*a', 'a'),
                # Eccentricity (e) - Look for 'Eccentricity \(e\)' followed by a number
                (r'Eccentricity \(e\)\s*([\d.]+)', 'e'),
                # Inclination (i) - Look for 'Inclination \(i\)' followed by a number and '°'
                (r'Inclination \(i\)\s*([\d.]+)\s*°', 'i'),
                # Longitude of periastron (omega) - Note: image shows omega (ω) labeled as "Longitude of periastron"
                # This could be tricky as the code expects 'omega' but label says 'Longitude of periastron'.
                # Make sure the key matches what is expected downstream.
                # If the image implies 'omega' is the 'Longitude of periastron', we'll extract it as 'omega'.
                (r'Longitude of periastron \(ω\)\s*([\d.]+)\s*°', 'omega'),
                # Node (Omega) - Look for 'Node (Ω)' followed by a number and '°'
                (r'Node \(Ω\)\s*([\d.]+)\s*°', 'Omega'),
            ]
            
            # Apply patterns
            for pattern_str, field in patterns:
                match = re.search(pattern_str, text, re.IGNORECASE)
                if match:
                    try:
                        orbital_elements[field] = float(match.group(1))
                    except ValueError:
                        # Continue if conversion to float fails for some reason
                        continue
            
            return orbital_elements
            
        except Exception as e:
            # Log the error for debugging if needed, but return current state
            print(f"Error extracting orbital elements: {e}")
            return orbital_elements
    
    async def get_wds_data(self, wds_id: str) -> Dict[str, Any]:
        """Get WDS data from Stelle Doppie website."""
        html_content = await self._fetch_star_page(wds_id)
        if html_content:
            return self._extract_star_data(html_content, wds_id)
        return {}
    
    async def get_orbital_elements(self, wds_id: str) -> Dict[str, Any]:
        """Get orbital elements from Stelle Doppie website."""
        html_content = await self._fetch_star_page(wds_id)
        if html_content:
            return self._extract_orbital_elements(html_content)
        return {}
    
    async def validate_physicality(self, wds_id: str) -> str:
        """Web source cannot validate physicality - returns Unknown."""
        return "Unknown"