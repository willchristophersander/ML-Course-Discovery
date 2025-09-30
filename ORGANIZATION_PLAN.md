# Project Organization Plan

## Current Issues
1. **Root directory clutter** - 50+ files in root directory
2. **Mixed file types** - Scripts, data, models, docs all mixed together
3. **Inconsistent naming** - Some files have timestamps, others don't
4. **Duplicate files** - Multiple versions of similar functionality
5. **Missing organization** - Good src/ structure exists but not used consistently

## Proposed Organization Structure

```
AppleAppSander/
├── src/                           # Main source code
│   ├── core/                      # Core system components
│   │   ├── __init__.py
│   │   ├── integrated_system.py   # Main integrated system
│   │   ├── course_discovery.py    # Course discovery logic
│   │   └── validation.py          # Validation components
│   ├── models/                    # ML models and classifiers
│   │   ├── __init__.py
│   │   ├── course_search_classifier.py
│   │   ├── course_page_classifier.py
│   │   ├── search_navigator.py
│   │   └── enhanced_classifier.py
│   ├── scripts/                   # Execution scripts
│   │   ├── __init__.py
│   │   ├── run_discovery.py       # Main execution script
│   │   ├── run_validation.py      # Validation script
│   │   ├── run_training.py        # Training script
│   │   └── test_system.py         # Testing script
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Data loading utilities
│   │   ├── file_utils.py          # File operations
│   │   └── analysis.py            # Analysis utilities
│   └── data/                      # Data files
│       ├── universities.json       # University database
│       ├── training_data.json     # Training datasets
│       └── validation_data.json   # Validation datasets
├── models/                        # Trained model files (.pkl)
│   ├── course_search_classifier.pkl
│   ├── course_page_classifier.pkl
│   ├── search_navigator.pkl
│   └── enhanced_classifier.pkl
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
│   ├── test_discovery.py
│   ├── test_validation.py
│   └── test_models.py
├── config/                        # Configuration files
│   ├── settings.json
│   └── logging.conf
├── requirements.txt               # Dependencies
├── README.md                     # Main README
└── .gitignore                    # Git ignore file
```

## File Migration Plan

### Root Directory Cleanup
**Move to src/core/:**
- `integrated_course_discovery_system.py` → `src/core/integrated_system.py`
- `google_course_search_finder_v2.py` → `src/core/course_discovery.py`
- `google_course_search_finder.py` → `src/core/course_discovery_v1.py`
- `simplecollegetransfercoursesearch.py` → `src/core/college_transfer_search.py`
- `collegetransferscript.py` → `src/core/college_transfer_v1.py`

**Move to src/scripts/:**
- `fast_discovery_validation.py` → `src/scripts/run_validation.py`
- `discover_and_validate_more_pages.py` → `src/scripts/discover_pages.py`
- `validate_discovered_pages.py` → `src/scripts/validate_pages.py`
- `test_integrated_system.py` → `tests/test_integrated_system.py`
- `test_google_course_finder.py` → `tests/test_google_finder.py`

**Move to models/:**
- `*.pkl` files → `models/`
- `course_search_classifier.pkl`
- `course_page_classifier.pkl`
- `page_navigation_classifier.pkl`
- `search_result_classifier.pkl`
- `page_navigation_classifier_v2.pkl`
- `search_result_classifier_v2.pkl`

**Move to results/:**
- `*_courses_*.json` files → `results/discovery_results/`
- `*_validation_*.json` files → `results/validation_results/`
- `*_training_*.json` files → `results/training_results/`

**Move to docs/:**
- `*_SUMMARY.md` files → `docs/summaries/`
- `*_GUIDE.md` files → `docs/guides/`
- `TECHNICAL_WRITEUP.md` → `docs/technical/`
- `UPDATED_COMMANDS.md` → `docs/guides/`

**Move to src/data/:**
- `world_universities_and_domains_and_courses.json` → `src/data/universities.json`
- `positive_course_catalogs.json` → `src/data/positive_examples.json`
- `negative_examples.json` → `src/data/negative_examples.json`
- `course_search_training_data.json` → `src/data/training_data.json`

**Delete/Archive:**
- `sample1.txt`, `sample2.txt` (empty files)
- `berkeley_test.png` (test image)
- `search_results.png` (test image)
- `record_and_generate_code.py` (development script)
- `record_college_transfer_search.py` (development script)

## Implementation Steps

1. **Create new directory structure**
2. **Move files to appropriate locations**
3. **Update import statements**
4. **Create __init__.py files**
5. **Update documentation references**
6. **Test system functionality**
7. **Clean up remaining files**

## Benefits

1. **Clear separation of concerns** - Code, data, models, docs separated
2. **Easier navigation** - Logical file organization
3. **Better maintainability** - Related files grouped together
4. **Cleaner root directory** - Only essential files at top level
5. **Consistent naming** - Standardized file naming conventions
6. **Version control friendly** - Better git organization 