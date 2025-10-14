# AppleAppSander - ML Course Catalog Finder

A machine learning system for discovering course search pages from college catalog pages using multi-model classification and Playwright-powered navigation. An introductory project to machine learning I pursued in my freetime -- as I learn more I plan to comeback to this project and implement what I've learned since.

##  Project Structure

```
AppleAppSander/
├── src/                           # Main source code
│   ├── core/                      # Core system components
│   │   ├── integrated_system.py   # Main integrated system
│   │   ├── course_discovery.py    # Course discovery logic
│   │   ├── college_transfer_search.py # CollegeTransfer.net integration
│   │   └── college_transfer_v1.py # Legacy college transfer script
│   ├── models/                    # ML models and classifiers
│   │   ├── course_search_classifier.py
│   │   ├── course_page_classifier.py
│   │   ├── search_navigator.py
│   │   └── enhanced_classifier.py
│   ├── scripts/                   # Execution scripts
│   │   ├── run_discovery.py       # Main execution script
│   │   ├── run_validation.py      # Validation script
│   │   ├── discover_pages.py      # Page discovery script
│   │   └── validate_pages.py      # Page validation script
│   ├── utils/                     # Utility functions
│   └── data/                      # Data files
│       ├── universities.json       # University database
│       ├── training_data.json     # Training datasets
│       ├── positive_examples.json # Positive training examples
│       └── negative_examples.json # Negative training examples
├── models/                        # Trained model files (.pkl)
├── results/                       # Output results
│   ├── discovery_results/         # Discovery outputs
│   ├── validation_results/        # Validation outputs
│   ├── training_results/          # Training outputs
│   └── analysis_results/          # Analysis outputs
├── docs/                          # Documentation
│   ├── technical/                 # Technical documentation
│   ├── guides/                    # User guides
│   └── summaries/                 # Project summaries
├── tests/                         # Test files
├── config/                        # Configuration files
│   └── settings.json              # System configuration
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

##  Quick Start

### **Setup Environment**
```bash
# Create virtual environment and install dependencies
python setup_environment.py
# Activate virtual environment
python source .venv/bin/activate
```

### **Run Course Discovery System**
```bash
# Navigate to scripts directory
cd src/scripts

# Run the main discovery system
python run_discovery.py
```

### **Run Validation**
```bash
# Run validation script
python run_validation.py
```

### **Run Page Discovery**
```bash
# Run page discovery script
python discover_pages.py
```

##  System Architecture

### **Core Components**
- **IntegratedCourseDiscoverySystem**: Main system orchestrator
- **EnhancedCourseSearchFinder**: Course search page discovery
- **CollegeTransferSearch**: Course validation via CollegeTransfer.net

### **Multi-Model Classification & Navigation**
- **SERP Scoring**: RandomForest ranks search results before navigation
- **Link Prioritisation**: RandomForest classifiers score internal links during crawling
- **Page Classification**: RandomForest/GradientBoosting models evaluate course search interfaces
- **Navigation Engine**: Playwright automates traversal with heuristic fallbacks and rate limiting

##  Training Data

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

##  Configuration

The system can be configured via `config/settings.json`:

```json
{
    "system": {
        "max_universities": 10,
        "max_test_courses": 5,
        "discovery_timeout": 30,
        "validation_timeout": 15,
        "request_delay": 2
    }
}
```

##  Results

### **Performance Metrics**
- **Success Rate**: Percentage of successful course search discoveries
- **Time Efficiency**: Average time to find course search page
- **Path Length**: Number of clicks to reach course search
- **Accuracy**: Precision/recall of course search detection

### **Validation**
- **Basic Validation**: Checks for course search functionality
- **Form Detection**: Identifies search forms and input fields
- **Keyword Analysis**: Course-related term frequency
- **Link Analysis**: Navigation pattern recognition

##  Development

### **Adding New Models**
```bash
cd src/models
# Create new model file
python your_new_model.py
```

### **Adding New Scripts**
```bash
cd src/scripts
# Create new script file
python your_new_script.py
```

### **Running Tests**
```bash
cd tests
python test_integrated_system.py
python test_google_finder.py
```

##  Documentation

- **`docs/technical/`**: Detailed technical architecture
- **`docs/guides/`**: User guides and commands
- **`docs/summaries/`**: Project summaries and status reports

##  Key Features

- **Multi-Modal ML**: Combines text, structure, and URL features
- **Automated Navigation**: Playwright-based crawling guided by trained classifiers
- **Parallel Processing**: Optimized for multi-core systems
- **From-Scratch Training**: No pre-trained models, custom-built
- **Scalable Architecture**: Easy to extend and modify
- **Organized Structure**: Clean, maintainable codebase
