import json
import random
import os

def load_existing_course_catalogs():
    """Load existing course catalog results to check for high confidence URLs"""
    existing_catalogs = {}
    
    # Try to load from multiple_universities_results.json
    try:
        with open('multiple_universities_results.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            if 'found_course_catalogs' in data:
                for catalog in data['found_course_catalogs']:
                    if catalog.get('confidence', 0) >= 0.7:  # High confidence threshold
                        existing_catalogs[catalog['university']] = {
                            'url': catalog['course_catalog'],
                            'confidence': catalog['confidence']
                        }
    except FileNotFoundError:
        print("No existing course catalog results found.")
    except Exception as e:
        print(f"Error loading existing results: {e}")
    
    # Try to load from world_universities_and_domains_and_courses.json
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'world_universities_and_domains_and_courses.json')
        with open(data_path, 'r', encoding='utf-8') as file:
            universities = json.load(file)
            for university in universities:
                if isinstance(university, dict):
                    name = university.get('name', '')
                    course_catalog = university.get('course_catalog')
                    if course_catalog and course_catalog != 'null' and course_catalog != None:
                        # If it has a course catalog URL, consider it as having high confidence
                        existing_catalogs[name] = {
                            'url': course_catalog,
                            'confidence': 0.8  # Assume high confidence for existing entries
                        }
    except FileNotFoundError:
        print("No world_universities_and_domains_and_courses.json found.")
    except Exception as e:
        print(f"Error loading world universities data: {e}")
    
    return existing_catalogs

def select_us_universities(num_colleges=20):
    """Read the JSON file and randomly select U.S. universities, skipping those with existing high confidence course catalogs"""
    
    # Load existing course catalogs
    existing_catalogs = load_existing_course_catalogs()
    print(f"Found {len(existing_catalogs)} universities with existing high confidence course catalogs")
    
    # Read the JSON file
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'world_universities_and_domains.json')
    with open(data_path, 'r', encoding='utf-8') as file:
        universities = json.load(file)
    
    # Filter for U.S. universities - only those explicitly marked as United States
    us_universities = []
    
    for university in universities:
        if isinstance(university, dict):
            country = university.get('country', '')
            if country == 'United States':
                us_universities.append(university)
    
    # Remove duplicates and skip universities with existing high confidence course catalogs
    unique_us_universities = []
    seen_names = set()
    skipped_count = 0
    
    for university in us_universities:
        name = university.get('name', '')
        if name not in seen_names:
            # Check if this university already has a high confidence course catalog
            if name in existing_catalogs:
                print(f"Skipping {name} - already has course catalog: {existing_catalogs[name]['url']}")
                skipped_count += 1
            else:
                unique_us_universities.append(university)
            seen_names.add(name)
    
    print(f"Found {len(unique_us_universities)} unique U.S. universities without existing course catalogs")
    print(f"Skipped {skipped_count} universities with existing course catalogs")
    
    # Randomly select universities from the remaining ones
    if len(unique_us_universities) >= num_colleges:
        selected = random.sample(unique_us_universities, num_colleges)
    else:
        print(f"Only found {len(unique_us_universities)} U.S. universities without existing course catalogs, selecting all of them")
        selected = unique_us_universities
    
    return selected

def main():
    print("Selecting 20 random U.S. universities (skipping those with existing high confidence course catalogs)...")
    selected_universities = select_us_universities()
    
    print(f"\nSelected {len(selected_universities)} U.S. universities:\n")
    
    for i, university in enumerate(selected_universities, 1):
        name = university.get('name', 'Unknown')
        domains = university.get('domains', [])
        country = university.get('country', 'Unknown')
        
        print(f"{i:2d}. {name}")
        print(f"    Country: {country}")
        if domains:
            print(f"    Domains: {', '.join(domains)}")
        print()

if __name__ == "__main__":
    main() 