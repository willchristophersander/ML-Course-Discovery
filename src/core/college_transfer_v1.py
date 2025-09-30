#!/usr/bin/env python3
"""
CollegeTransfer.net automation script using Python Playwright
"""

import asyncio
import json
import re
import sys
import os
import argparse
from datetime import datetime
from playwright.async_api import async_playwright

def generate_university_name_variations(university_name):
    """Generate different variations of university names to try"""
    variations = [university_name]  # Start with original name
    
    # Common transformations
    if ',' in university_name:
        # Try replacing comma with dash
        variations.append(university_name.replace(',', '-'))
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

async def search_with_variations(page, university_name):
    """Try searching with different university name variations"""
    variations = generate_university_name_variations(university_name)
    
    for i, variation in enumerate(variations):
        print(f" Trying variation {i+1}/{len(variations)}: '{variation}'")
        
        try:
            # Navigate to the course search page
            await page.goto('https://www.collegetransfer.net/Search/Search-for-Courses')
            
            # Wait until search area loads
            await page.wait_for_selector('text=By Keyword')
            
            # Click on "By Keyword" tab
            await page.click('text=By Keyword')
            
            # Wait for and click on the institution input field
            await page.wait_for_selector('input#dnn_ctr481_StudentAdvanceSearch_ucCourseSearchByKeywordPanel_ctl00_ucSearchByKeyword_ucInstitutionName_ucInstitutionTypeAhead_txtTypeAhead')
            await page.click('input#dnn_ctr481_StudentAdvanceSearch_ucCourseSearchByKeywordPanel_ctl00_ucSearchByKeyword_ucInstitutionName_ucInstitutionTypeAhead_txtTypeAhead')
            
            # Fill in the university name variation
            await page.fill('input#dnn_ctr481_StudentAdvanceSearch_ucCourseSearchByKeywordPanel_ctl00_ucSearchByKeyword_ucInstitutionName_ucInstitutionTypeAhead_txtTypeAhead', variation)
            
            # Try to click on suggestion if available, otherwise continue without it
            try:
                await page.wait_for_selector('mark', timeout=3000)
                await page.click('mark')
                print(f"   Found suggestion for '{variation}'")
            except:
                print(f"    No suggestion found for '{variation}', trying search anyway")
                # Press Tab to move to next field
                await page.keyboard.press('Tab')
            
            # Wait a moment for the form to be ready
            await page.wait_for_timeout(1000)
            
            # Click the search button using JavaScript
            await page.evaluate("document.getElementById('dnn_ctr481_StudentAdvanceSearch_ucCourseSearchByKeywordPanel_ctl01_btnKeywordSearch').click()")
            
            # Wait for navigation to results page
            try:
                await page.wait_for_url('**/Course-Search-Results**', timeout=10000)
                
                # Check if we got results
                await page.wait_for_load_state('networkidle')
                
                # Extract course data
                courses = await page.evaluate("""
                    (universityName) => {
                        const courses = [];
                        
                        // Get the full page text content
                        const pageText = document.body.textContent || '';
                        
                        // Split by lines and process
                        const lines = pageText.split('\\n').map(line => line.trim()).filter(line => line.length > 0);
                        
                        for (let i = 0; i < lines.length; i++) {
                            const line = lines[i];
                            
                            // Look for course code patterns like "HST 014:", "POLS 168:", etc.
                            const courseMatch = line.match(/^([A-Z]{2,4}\\s+\\d{3}):\\s*(.+)$/);
                            
                            if (courseMatch) {
                                const courseCode = courseMatch[1].trim();
                                const courseTitle = courseMatch[2].trim();
                                
                                let credits = '';
                                let description = '';
                                
                                // Look ahead for credits and description
                                let j = i + 1;
                                let descriptionStart = -1;
                                
                                while (j < lines.length && j < i + 20) { // Look up to 20 lines ahead
                                    const nextLine = lines[j];
                                    
                                    // Stop if we hit another course
                                    if (nextLine.match(/^[A-Z]{2,4}\\s+\\d{3}:/)) {
                                        break;
                                    }
                                    
                                    // Look for credits
                                    if (!credits && nextLine.match(/(\\d+\\.\\d+)\\s*Credits/)) {
                                        credits = nextLine.match(/(\\d+\\.\\d+)\\s*Credits/)[1];
                                    }
                                    
                                    // Look for description start (usually after credits and institution info)
                                    if (nextLine.includes('Credits:') && !descriptionStart) {
                                        descriptionStart = j + 1;
                                    }
                                    
                                    // If we found description start, collect description lines
                                    if (descriptionStart !== -1 && j >= descriptionStart) {
                                        // Skip navigation elements
                                        if (!nextLine.includes('Share') && 
                                            !nextLine.includes('Favorite') && 
                                            !nextLine.includes('Show comparable courses') &&
                                            !nextLine.includes('University of Vermont') &&
                                            !nextLine.includes('External link') &&
                                            !nextLine.includes('Opens in a new window') &&
                                            nextLine.length > 5) {
                                            description += nextLine + ' ';
                                        }
                                    }
                                    
                                    j++;
                                }
                                
                                // Clean up description
                                description = description.trim();
                                
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
                        }
                        
                        return courses;
                    }
                """, university_name)  # Use original name for institution field
                
                if len(courses) > 0:
                    print(f" Found {len(courses)} courses with variation: '{variation}'")
                    return courses, variation
                else:
                    print(f" No courses found with variation: '{variation}'")
                    
            except Exception as e:
                print(f"    Error during search with variation '{variation}': {str(e)}")
                continue
                
        except Exception as e:
            print(f"  Error navigating with variation '{variation}': {str(e)}")
            continue
    
    # If we get here, no variations worked
    print(f" No courses found with any variation of '{university_name}'")
    return [], university_name

async def automate_college_transfer(university_name):
    """Automate CollegeTransfer.net course search for a specific university"""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print(f' Searching for courses at: {university_name}')
        
        # Try searching with different name variations
        courses, successful_variation = await search_with_variations(page, university_name)
        
        if len(courses) == 0:
            print(" No courses found with any university name variation")
            await browser.close()
            return
        
        print(f' Search completed successfully!')
        print(f' Successful variation: "{successful_variation}"')
        
        # Save courses to JSON file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_university_name = university_name.replace(' ', '_').replace(',', '').replace('.', '')
        
        # Create course_data directory if it doesn't exist
        course_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'course_data')
        os.makedirs(course_data_dir, exist_ok=True)
        
        filename = os.path.join(course_data_dir, f'{safe_university_name}_courses_{timestamp}.json')
        
        with open(filename, 'w') as f:
            json.dump({
                'search_query': {
                    'institution': university_name,
                    'keyword': '',  # No subject keyword
                    'successful_variation': successful_variation,
                    'url': page.url
                },
                'total_results': len(courses),
                'courses': courses,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f' Found {len(courses)} course results')
        print(f' Course data saved to: {filename}')
        
        # Display first few courses as preview
        if courses:
            print('\n Sample courses:')
            for i, course in enumerate(courses[:5]):
                print(f'  {i+1}. {course["course_id"]} - {course["course_title"]}')
                if course.get("credits"):
                    print(f'     Credits: {course["credits"]}')
                if course.get("description"):
                    print(f'     Description: {course["description"][:200]}...')
                print()
            if len(courses) > 5:
                print(f'  ... and {len(courses) - 5} more courses')
        
        # Take a screenshot for verification
        try:
            screenshot_path = os.path.join(course_data_dir, 'search_results.png')
            await page.screenshot(path=screenshot_path)
            print(f' Screenshot saved as {screenshot_path}')
        except Exception as e:
            print(f'  Could not take screenshot: {e}')
        
        await browser.close()

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Search for courses at a specific university on CollegeTransfer.net')
    parser.add_argument('university', help='Name of the university to search for (e.g., "University of Vermont")')
    
    args = parser.parse_args()
    
    await automate_college_transfer(args.university)

if __name__ == "__main__":
    asyncio.run(main())