# Technical Writeup: ML Course Catalog Finder

##  **System Architecture**

### **Project Structure**
```
AppleAppSander/
├── src/
│   ├── models/                      # ML models and classifiers
│   │   ├── simple_course_search_classifier.py    # Course search classifier
│   │   └── course_page_classifier.py             # Course page classifier
│   ├── scripts/                     # Main execution scripts
│   │   ├── ml_course_catalog_finder.py          # Core ML pipeline
│   │   ├── run_comprehensive_course_search.py    # Main execution script
│   │   ├── parallel_genetic_trainer.py          # Parallel genetic training
│   │   └── genetic_link_discovery_trainer.py    # Genetic algorithm trainer
│   ├── data/                        # Data files
│   │   ├── world_universities_and_domains.json
│   │   └── world_universities_and_domains_and_courses.json
│   └── utils/                       # Utility functions
├── models/                          # Trained model files
├── results/                         # Training results and outputs
├── docs/                           # Documentation
└── venv/                           # Python virtual environment
```

##  **Model Architecture**

### **Multi-Modal Classification System**

The system uses a **Random Forest Classifier** that combines multiple types of features:

#### **1. Text Features**
- **Course-related keywords**: "course", "class", "schedule", "catalog", "search", "register"
- **Search interface terms**: "search courses", "find classes", "browse catalog"
- **Academic terms**: "department", "subject", "credits", "prerequisites"

#### **2. Structural Features**
- **Form elements**: Search forms, input fields, submit buttons
- **Navigation patterns**: Course-related links, breadcrumbs
- **Page structure**: Tables, lists, content sections

#### **3. URL Features**
- **Domain patterns**: `classes.domain.edu`, `courses.domain.edu`
- **Path analysis**: `/course-search/`, `/class-schedule/`
- **Query parameters**: Search-related URL parameters

### **Feature Engineering**

```python
def extract_simple_features(self, soup, url):
    """Extract features from HTML content and URL."""
    features = {}
    
    # Text-based features
    page_text = soup.get_text().lower()
    features['course_keywords'] = sum(1 for word in ['course', 'class', 'schedule'] if word in page_text)
    features['search_keywords'] = sum(1 for word in ['search', 'find', 'browse'] if word in page_text)
    
    # Structural features
    features['total_forms'] = len(soup.find_all('form'))
    features['total_inputs'] = len(soup.find_all('input'))
    features['total_links'] = len(soup.find_all('a'))
    
    # URL features
    features['has_course_in_url'] = 'course' in url.lower()
    features['has_class_in_url'] = 'class' in url.lower()
    
    return features
```

##  **Genetic Algorithm**

### **NavigationAgent Class**

Each agent represents a navigation strategy with these parameters:

```python
@dataclass
class NavigationAgent:
    max_depth: int                    # How deep to follow links (3-5)
    max_links_per_page: int          # Links to explore per page (10-20)
    link_preference_weights: List[float]  # Preference for different link types
    patience_factor: float           # Time between requests (2-5 seconds)
    exploration_rate: float          # Random exploration vs. guided search
```

### **Fitness Function**

Agents are evaluated based on:
- **Success rate**: Percentage of successful course search discoveries
- **Time efficiency**: Average time to find course search page
- **Path length**: Number of clicks to reach course search
- **Accuracy**: Precision/recall of course search detection

### **Genetic Operators**

#### **Mutation**
```python
def mutate_agent(self, agent):
    """Apply random mutations to agent parameters."""
    if random.random() < 0.3:
        agent.max_depth = max(1, min(10, agent.max_depth + random.randint(-1, 1)))
    if random.random() < 0.3:
        agent.max_links_per_page = max(5, min(50, agent.max_links_per_page + random.randint(-5, 5)))
    # ... more mutations
```

#### **Crossover**
```python
def crossover_agents(self, parent1, parent2):
    """Combine two parent agents to create offspring."""
    child = NavigationAgent(
        max_depth=random.choice([parent1.max_depth, parent2.max_depth]),
        max_links_per_page=random.choice([parent1.max_links_per_page, parent2.max_links_per_page]),
        # ... combine other parameters
    )
    return child
```

##  **Training Process**

### **1. Data Collection**
- **Positive examples**: Known course search pages
- **Negative examples**: Catalog/bulletin pages
- **Training data**: 8 course search pages, 9 catalog pages

### **2. Model Training**
```python
# Train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
```

### **3. Genetic Evolution**
```python
# Initialize population
population = [NavigationAgent() for _ in range(population_size)]

for generation in range(generations):
    # Evaluate fitness
    fitness_scores = [evaluate_agent(agent) for agent in population]
    
    # Selection
    parents = tournament_selection(population, fitness_scores)
    
    # Crossover and mutation
    offspring = []
    for _ in range(population_size // 2):
        parent1, parent2 = random.sample(parents, 2)
        child1 = crossover_agents(parent1, parent2)
        child2 = crossover_agents(parent1, parent2)
        offspring.extend([mutate_agent(child1), mutate_agent(child2)])
    
    # Elitism: keep best agents
    population = select_best_agents(population, fitness_scores, elitism_rate) + offspring
```

##  **Training Data**

### **Course Search Pages** (Positive Examples)
- UC Berkeley Class Schedule: `https://classes.berkeley.edu/`
- CUNY Global Class Search: `https://globalsearch.cuny.edu/`
- Ohio State University Class Search: `https://classes.osu.edu/`
- University of Illinois Course Explorer: `https://go.illinois.edu/courseexplorer`
- University of New Hampshire Courses: `https://courses.unh.edu/`
- University of Chicago Class Search: `https://coursesearch.uchicago.edu/`
- Harvard Course Search: `https://courses.my.harvard.edu/psp/courses/EMPLOYEE/EMPL/h/?tab=HU_CLASS_SEARCH`
- University of Houston Class Search: `https://saprd.my.uh.edu/psc/saprd/EMPLOYEE/HRMS/c/COMMUNITY_ACCESS.CLASS_SEARCH.GBL`

### **Catalog Pages** (Negative Examples)
- University of Vermont Undergraduate Catalog: `https://catalogue.uvm.edu/undergraduate/`
- University of Utah General Catalog: `https://registrar.utah.edu/Catalog-schedules.php`
- UNC Chapel Hill Catalog: `https://catalog.unc.edu/`
- UT Austin Catalog: `https://catalog.utexas.edu/`
- University of Kentucky Catalog: `https://catalogs.uky.edu/`
- College of Charleston Catalog: `https://catalog.cofc.edu/`
- University of Northern Iowa Catalog: `https://catalog.uni.edu/`
- Seattle Colleges Academic Catalog: `https://www.seattlecolleges.edu/academics/academic-catalog`
- Cal Poly Academic Catalog: `https://catalog.calpoly.edu/`

##  **Configuration Parameters**

### **Model Parameters**
- **Max Depth**: 3-5 levels of link following
- **Max Links Per Page**: 10-20 links to explore
- **Confidence Threshold**: 0.3-0.5 for course search detection
- **Patience Factor**: 2-5 seconds between requests

### **Genetic Algorithm Parameters**
- **Population Size**: 20-100 agents per generation
- **Generations**: 10-50 generations
- **Mutation Rate**: 0.1-0.3
- **Elitism**: Keep top 10% of agents

##  **Performance Metrics**

### **Success Rate**
- Percentage of successful course search discoveries
- Target: >50% success rate

### **Time Efficiency**
- Average time to find course search page
- Target: <30 seconds per university

### **Path Length**
- Number of clicks to reach course search
- Target: <5 clicks on average

### **Accuracy**
- Precision/recall of course search detection
- Target: >80% precision, >70% recall

##  **Validation System**

### **Basic Validation**
The system uses basic validation to check if a discovered page is a course search page:

```python
def validate_course_search_page(self, url, university_name=None):
    """Basic validation of a course search page."""
    # Check for course search indicators
    course_keywords = ['course', 'class', 'schedule', 'catalog', 'search', 'register']
    
    # Check for search forms and input fields
    search_forms = soup.find_all('form')
    input_fields = soup.find_all('input')
    
    # Calculate validation score
    validation_score = 0
    for keyword in course_keywords:
        if keyword in page_text:
            validation_score += 1
    
    if search_forms:
        validation_score += 2
    
    if len(input_fields) > 0:
        validation_score += 1
    
    # Return validation result
    is_valid = validation_score >= 5  # At least 50% score
    return {'valid': is_valid, 'score': validation_score}
```

##  **Execution Flow**

### **1. Initialization**
```python
# Load trained models
classifier = load_model('simple_course_search_classifier.pkl')

# Initialize genetic population
population = [NavigationAgent() for _ in range(population_size)]
```

### **2. Navigation Process**
```python
def navigate_with_agent(agent, start_url):
    """Navigate from catalog page to course search page."""
    visited_urls = set()
    queue = [(start_url, 0)]  # (url, depth)
    
    while queue:
        current_url, depth = queue.pop(0)
        
        if depth > agent.max_depth:
            continue
            
        # Fetch page
        soup = fetch_page(current_url)
        
        # Check if this is a course search page
        if classifier.predict([extract_features(soup, current_url)])[0] == 1:
            return current_url, True
        
        # Find links to follow
        links = find_links(soup, agent.link_preference_weights)
        
        for link in links[:agent.max_links_per_page]:
            if link not in visited_urls:
                visited_urls.add(link)
                queue.append((link, depth + 1))
    
    return None, False
```

### **3. Fitness Evaluation**
```python
def evaluate_agent(agent):
    """Evaluate agent's fitness on training data."""
    successes = 0
    total_time = 0
    
    for catalog_url, target_search_url in training_data:
        start_time = time.time()
        found_url, success = navigate_with_agent(agent, catalog_url)
        end_time = time.time()
        
        if success:
            successes += 1
            total_time += (end_time - start_time)
    
    success_rate = successes / len(training_data)
    avg_time = total_time / max(successes, 1)
    
    return success_rate * 0.7 + (1 / avg_time) * 0.3  # Weighted fitness
```

##  **Key Innovations**

### **1. Multi-Modal Feature Engineering**
- Combines text, structure, and URL features
- Robust to different page layouts and designs

### **2. Genetic Navigation Evolution**
- Agents evolve navigation strategies
- Adapts to different university website structures

### **3. Parallel Processing**
- Optimized for multi-core systems (Apple Silicon M1)
- Scales to large populations and datasets

### **4. From-Scratch Training**
- No pre-trained models or transfer learning
- Custom-built for course search detection

##  **Type Call Sequence**

### **Complete Pipeline Example**
```python
# 1. Initialize system
finder = MLCourseCatalogFinder()

# 2. Find course catalog for university
result = finder.find_course_catalog(
    university_name="University of California Berkeley",
    domain="berkeley.edu"
)

# 3. Result contains:
# - success: True/False
# - course_search_url: Found URL or None
# - path: List of URLs followed
# - time_taken: Seconds to find
# - confidence: ML model confidence score
```

### **Genetic Training Example**
```python
# 1. Initialize genetic trainer
trainer = ParallelGeneticTrainer(
    population_size=50,
    generations=10,
    parallel=True
)

# 2. Run training
results = trainer.train()

# 3. Results contain:
# - best_agent: Best performing navigation agent
# - fitness_history: Fitness scores over generations
# - success_rate: Final success rate
# - avg_time: Average time to find course search
```

##  **Summary**

This system represents a **complete ML pipeline** for discovering course search pages from college catalog pages using:

- **Multi-modal classification** with Random Forest
- **Genetic algorithm evolution** for navigation strategies
- **Parallel processing** for efficient training
- **From-scratch training** with custom features
- **Basic validation** for course search detection

The system is designed to be **scalable**, **robust**, and **adaptable** to different university website structures and layouts. 