#!/usr/bin/env python3
"""CollegeTransfer.net course search utilities using a shared Playwright client."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

try:
    from core.collegetransfer_client import CollegeTransferClient  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when executed as package module
    from .collegetransfer_client import CollegeTransferClient


COURSE_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "course_data"
COURSE_DATA_DIR.mkdir(parents=True, exist_ok=True)

def format_university_name_for_url(university_name):
    """
    Format university name for CollegeTransfer.net URL
    Converts spaces to '+' for URL encoding
    """
    # Replace spaces with '+' for URL encoding
    formatted_name = university_name.replace(' ', '+')
    return formatted_name

def generate_university_name_variations(university_name):
    """Generate different variations of university names to try"""
    variations = [university_name]  # Start with original name
    
    # Common transformations
    if ',' in university_name:
        # Try replacing comma with dash (remove space after comma)
        variations.append(university_name.replace(', ', '-'))
        # Try removing comma and space
        variations.append(university_name.replace(', ', ''))
    
    if '-' in university_name:
        # Try replacing dash with comma
        variations.append(university_name.replace('-', ', '))
    
    # Try common abbreviations
    if 'University of' in university_name:
        variations.append(university_name.replace('University of', 'U of'))
        variations.append(university_name.replace('University of', 'Univ of'))
    
    if 'University' in university_name:
        variations.append(university_name.replace('University', 'Univ'))
    
    # Try removing "University" entirely for some cases
    if university_name.startswith('University of '):
        variations.append(university_name.replace('University of ', ''))
    
    # Try adding "University" if not present
    if 'University' not in university_name and 'College' not in university_name:
        variations.append(f"University of {university_name}")
    
    # Special cases for UC Berkeley
    if 'California, Berkeley' in university_name or 'California-Berkeley' in university_name:
        variations.extend([
            'UC Berkeley',
            'University of California Berkeley',
            'University of California-Berkeley',
            'Berkeley',
            'UCB'
        ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for variation in variations:
        if variation not in seen:
            seen.add(variation)
            unique_variations.append(variation)
    
    return unique_variations

def generate_collegetransfer_url(university_name):
    """
    Generate CollegeTransfer.net search URL for a university
    """
    formatted_name = format_university_name_for_url(university_name)
    base_url = "https://www.collegetransfer.net/Search/Search-for-Courses/Course-Search-Results"
    url = f"{base_url}?instnm={formatted_name}&distance=5"
    return url

async def scrape_course_results(page, university_name):
    """
    Scrape course results from the CollegeTransfer.net page using HTML element extraction
    """
    try:
        # Wait for the page to load
        await page.wait_for_load_state('networkidle')
        
        # Extract course data using direct HTML element selection
        courses = await page.evaluate("""
            (universityName) => {
                const courses = [];
                
                // Use the specific CSS selector to find course result elements
                const courseElements = document.querySelectorAll('.student-course-search-results .student-course-search-results-list li');
                
                courseElements.forEach(element => {
                    try {
                        // Get the full text content of this course element
                        const elementText = element.textContent || '';
                        const lines = elementText.split('\\n')
                            .map(line => line.trim())
                            .filter(line => line.length > 0);
                        
                        let courseCode = '';
                        let courseTitle = '';
                        let credits = '';
                        let description = '';
                        
                        // Process each line in the course element
                        for (let i = 0; i < lines.length; i++) {
                            const line = lines[i];
                            
                            // Look for course code patterns like "ACCT 301:", "HST 014:", "AEROSPC 135A:", etc.
                            const courseMatch = line.match(/^([A-Z]{2,8}\\s+\\d{3}[A-Z]?):\\s*(.+)$/);
                            
                            if (courseMatch) {
                                courseCode = courseMatch[1].trim();
                                courseTitle = courseMatch[2].trim();
                                
                                // Look ahead for credits and description
                                let j = i + 1;
                                let foundCredits = false;
                                let descriptionLines = [];
                                
                                while (j < lines.length) {
                                    const nextLine = lines[j];
                                    
                                    // Stop if we hit another course code
                                    if (nextLine.match(/^[A-Z]{2,8}\\s+\\d{3}[A-Z]?:/)) {
                                        break;
                                    }
                                    
                                    // Look for credits
                                    if (!credits && nextLine.match(/(\\d+\\.\\d+)\\s*Credits/)) {
                                        credits = nextLine.match(/(\\d+\\.\\d+)\\s*Credits/)[1];
                                        foundCredits = true;
                                        j++;
                                        continue;
                                    }
                                    
                                    // Collect description after credits
                                    if (foundCredits) {
                                        // Skip navigation elements and UI text
                                        if (!nextLine.includes('Share') && 
                                            !nextLine.includes('Favorite') && 
                                            !nextLine.includes('Show comparable courses') &&
                                            !nextLine.includes('External link') &&
                                            !nextLine.includes('Opens in a new window') &&
                                            !nextLine.match(/(\\d+\\.\\d+)\\s*Credits/) &&
                                            nextLine.length > 3) {
                                            descriptionLines.push(nextLine);
                                        }
                                    }
                                    
                                    j++;
                                }
                                
                                // Join description lines
                                description = descriptionLines.join(' ').trim();
                                break; // Found course, exit the main loop
                            }
                        }
                        
                        // Only add course if we found a valid course code
                        if (courseCode && courseTitle) {
                            // Check if this course is already in our list
                            const existingCourse = courses.find(c => c.course_id === courseCode);
                            if (!existingCourse) {
                                courses.push({
                                    course_id: courseCode,
                                    course_title: courseTitle,
                                    credits: credits,
                                    description: description,
                                    institution: universityName
                                });
                            }
                        }
                        
                    } catch (error) {
                        console.error('Error processing course element:', error);
                    }
                });
                
                return courses;
            }
        """, university_name)
        
        return courses
        
    except Exception as e:
        print(f"Error scraping course results: {str(e)}")
        return []

async def search_with_variations(page, university_name):
    """Try searching with different university name variations"""
    variations = generate_university_name_variations(university_name)
    
    for i, variation in enumerate(variations):
        print(f" Trying variation {i+1}/{len(variations)}: '{variation}'")
        
        try:
            # Generate URL for this variation
            url = generate_collegetransfer_url(variation)
            
            # Navigate to the search results page
            await page.goto(url)
            
            # Wait for the page to load
            await page.wait_for_load_state('networkidle')
            
            # Check if we got results
            page_content = await page.content()
            if "No results found" in page_content or "no courses" in page_content.lower():
                print(f"   No courses found with variation: '{variation}'")
                continue
            
            # Extract course data
            courses = await scrape_course_results(page, university_name)  # Use original name for institution field
            
            if len(courses) > 0:
                print(f"   Found {len(courses)} courses with variation: '{variation}'")
                return courses, variation
            else:
                print(f"   No courses found with variation: '{variation}'")
                
        except Exception as e:
            print(f"    Error during search with variation '{variation}': {str(e)}")
            continue
    
    # If we get here, no variations worked
    print(f" No courses found with any variation of '{university_name}'")
    return [], university_name

async def search_university_courses(
    university_name: str,
    *,
    client: CollegeTransferClient | None = None,
    persist_results: bool = True,
) -> List[dict]:
    """Search for courses at a specific university and optionally persist results."""

    print(f" Searching for courses at: {university_name}")

    if client is None:
        async with CollegeTransferClient() as managed_client:
            return await _search_with_client(
                managed_client, university_name, persist_results=persist_results
            )

    return await _search_with_client(client, university_name, persist_results=persist_results)


async def _search_with_client(
    client: CollegeTransferClient,
    university_name: str,
    *,
    persist_results: bool = True,
) -> List[dict]:
    page = await client.ensure_ready()

    try:
        courses, successful_variation = await search_with_variations(page, university_name)

        if len(courses) == 0:
            print(" No courses found with any university name variation")
            return []

        print(" Search completed successfully!")
        print(f' Successful variation: "{successful_variation}"')

        if persist_results:
            _persist_course_results(university_name, successful_variation, courses)
            await _capture_screenshot(page)

        _print_preview(courses)
        return courses
    except Exception as exc:  # pragma: no cover - network variability
        print(f" Error during search: {exc}")
        return []


def _persist_course_results(university_name: str, variation: str, courses: List[dict]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_university_name = (
        university_name.replace(" ", "_").replace(",", "").replace(".", "")
    )
    filename = COURSE_DATA_DIR / f"{safe_university_name}_courses_{timestamp}.json"

    payload = {
        "search_query": {
            "institution": university_name,
            "successful_variation": variation,
            "url": generate_collegetransfer_url(variation),
        },
        "total_results": len(courses),
        "courses": courses,
        "timestamp": datetime.now().isoformat(),
    }

    with filename.open("w") as fh:
        json.dump(payload, fh, indent=2)

    print(f" Course data saved to: {filename}")
    return filename


async def _capture_screenshot(page) -> None:
    screenshot_path = COURSE_DATA_DIR / "search_results.png"
    try:
        await page.screenshot(path=str(screenshot_path))
        print(f" Screenshot saved as {screenshot_path}")
    except Exception as exc:  # pragma: no cover - optional best effort
        print(f"  Could not take screenshot: {exc}")


def _print_preview(courses: Iterable[dict]) -> None:
    courses = list(courses)
    if not courses:
        return

    print("\n Sample courses:")
    for i, course in enumerate(courses[:5]):
        print(f"  {i + 1}. {course.get('course_id', 'N/A')} - {course.get('course_title', 'Untitled')}")
        if course.get("credits"):
            print(f"     Credits: {course['credits']}")
        if course.get("description"):
            print(f"     Description: {course['description']}")
        print()

    remaining = len(courses) - 5
    if remaining > 0:
        print(f"  ... and {remaining} more courses")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python simplecollegetransfercoursesearch.py 'University Name'")
        print("Example: python simplecollegetransfercoursesearch.py 'University of Vermont'")
        sys.exit(1)
    
    university_name = sys.argv[1]
    
    # Run the async search
    asyncio.run(search_university_courses(university_name))

if __name__ == "__main__":
    main() 
