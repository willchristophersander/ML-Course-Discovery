# Updated Terminal Commands

##  Quick Start Commands

### **Setup Environment**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install requests beautifulsoup4 scikit-learn numpy pandas tqdm
```

### **Run Main Pipeline**
```bash
# Navigate to scripts
cd src/scripts

# Run comprehensive course search
python run_comprehensive_course_search.py 5 --parallel
```

### **Run Genetic Training**
```bash
# Navigate to scripts
cd src/scripts

# Run parallel genetic training
python parallel_genetic_trainer.py --generations 10 --agents 50 --parallel
```

##  Directory Structure

```
AppleAppSander/
├── src/
│   ├── models/                      # ML models and classifiers
│   ├── scripts/                     # Main execution scripts
│   ├── data/                        # Data files
│   └── utils/                       # Utility functions
├── models/                          # Trained model files
├── results/                         # Training results and outputs
├── docs/                           # Documentation
└── venv/                           # Python virtual environment
```

##  Development Commands

### **Test Models**
```bash
# Navigate to models
cd src/models

# Test course search classifier
python simple_course_search_classifier.py

# Test course page classifier
python course_page_classifier.py
```

### **Test Scripts**
```bash
# Navigate to scripts
cd src/scripts

# Test enhanced course search
python test_enhanced_course_search.py

# Test genetic trainer
python genetic_link_discovery_trainer.py
```

### **Run Genetic Training**
```bash
# Navigate to scripts
cd src/scripts

# Run parallel genetic training
python parallel_genetic_trainer.py --generations 10 --agents 50 --parallel

# Run diagnostic genetic training
python diagnostic_genetic_trainer.py
```

##  Data Management

### **Update University Database**
```bash
# Navigate to utils
cd src/utils

# Select US universities
python select_us_universities_final.py
```

### **Clean Up Files**
```bash
# Navigate to root
cd /Users/will/AppleAppSander

# Clean up unused files
python cleanup_unused_files.py
```

##  Model Training

### **Train Course Search Classifier**
```bash
# Navigate to models
cd src/models

# Train simple classifier
python simple_course_search_classifier.py
```

### **Train Course Page Classifier**
```bash
# Navigate to models
cd src/models

# Train page classifier
python course_page_classifier.py
```

##  Results and Outputs

### **View Results**
```bash
# Navigate to results
cd results

# List result files
ls -la
```

### **View Documentation**
```bash
# Navigate to docs
cd docs

# List documentation files
ls -la
```

##  Testing and Validation

### **Test Course Search Discovery**
```bash
# Navigate to scripts
cd src/scripts

# Test with specific university
python ml_course_catalog_finder.py
```

### **Test Genetic Algorithm**
```bash
# Navigate to scripts
cd src/scripts

# Test genetic training
python parallel_genetic_trainer.py --generations 1 --agents 10 --fast
```

##  Development Workflow

### **Create New Model**
```bash
# Navigate to models
cd src/models

# Create new model file
touch my_new_model.py
```

### **Create New Script**
```bash
# Navigate to scripts
cd src/scripts

# Create new script file
touch my_new_script.py
```

### **Test Import**
```bash
# Test model import
cd src/models && python -c "from simple_course_search_classifier import SimpleCourseSearchClassifier; print('Import successful')"

# Test script import
cd src/scripts && python -c "from ml_course_catalog_finder import MLCourseCatalogFinder; print('Import successful')"
```

##  Documentation

### **Update Documentation**
```bash
# Navigate to root
cd /Users/will/AppleAppSander

# Update README
# Edit README.md

# Update commands
# Edit UPDATED_COMMANDS.md
```

##  Key Features

- **Multi-Modal ML**: Combines text, structure, and URL features
- **Genetic Evolution**: Evolves navigation strategies
- **Parallel Processing**: Optimized for multi-core systems
- **From-Scratch Training**: No pre-trained models, custom-built
- **Scalable Architecture**: Easy to extend and modify

##  Performance Metrics

- **Success Rate**: Percentage of successful course search discoveries
- **Time Efficiency**: Average time to find course search page
- **Path Length**: Number of clicks to reach course search
- **Accuracy**: Precision/recall of course search detection

##  Configuration

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