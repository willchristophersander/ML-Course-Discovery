# Course Search Navigator Model

## Overview

The Course Search Navigator Model is an advanced machine learning system designed to find course search pages from initial college catalog pages. It uses a multi-modal approach combining content analysis, link relationship analysis, URL pattern analysis, and navigation hierarchy analysis.

## Architecture

### Multi-Modal Approach

The model combines several types of analysis:

1. **Content Analysis**
   - Text pattern matching for course search indicators
   - Form analysis (search inputs, dropdowns, buttons)
   - Export functionality detection
   - Results section identification
   - Advanced filtering detection

2. **Link Relationship Analysis**
   - Analysis of all links on the page
   - Scoring of links based on text and URL patterns
   - Navigation structure analysis
   - Identification of course-related navigation items

3. **URL Pattern Analysis**
   - URL path analysis
   - Subdomain analysis
   - Query parameter analysis
   - Pattern matching against known course search URLs

4. **Structural Analysis**
   - HTML structure analysis
   - Form structure analysis
   - Navigation hierarchy analysis

## Model Types & Recommendations

### 1. **Gradient Boosting Classifier (Recommended)**
- **Why**: Excellent for tabular data with mixed feature types
- **Advantage**: Handles non-linear relationships well, provides feature importance
- **Implementation**: Already implemented in the current model

### 2. **Graph Neural Network (Future Enhancement)**
- **Why**: College websites have hierarchical structures with navigation links
- **Advantage**: Can model the relationship between pages and their connections
- **Implementation**: Could be added using PyTorch Geometric or DGL

### 3. **Transformer-based Model (Future Enhancement)**
- **Why**: Can understand context and relationships in text
- **Advantage**: Pre-trained models like BERT can be fine-tuned
- **Implementation**: Could be added using Hugging Face transformers

### 4. **Ensemble Approach (Current Implementation)**
- **Why**: Combines the strengths of different approaches
- **Advantage**: More robust and interpretable
- **Implementation**: Currently uses Gradient Boosting with comprehensive feature engineering

## Key Features

### Content Analysis Features
- Search pattern matching (30+ patterns)
- Form element analysis
- Export functionality detection
- Results section identification
- Advanced filtering detection
- Course keyword density analysis

### Link Analysis Features
- Course search link identification
- Link text pattern matching
- Navigation structure analysis
- Link scoring based on multiple criteria

### URL Analysis Features
- URL pattern matching
- Path depth analysis
- Subdomain analysis
- Query parameter analysis

## Usage

### Training the Model

```bash
python train_course_search_navigator.py
```

The training script will:
1. Collect training data from university websites
2. Extract comprehensive features
3. Train the model
4. Evaluate performance
5. Save the trained model

### Using the Model

```python
from course_search_navigator_model import CourseSearchNavigatorModel

# Initialize the model
model = CourseSearchNavigatorModel()

# Load trained model
model.load_model()

# Predict if a page is a course search page
is_course_search, confidence, details = model.predict_course_search(url, html_content)

# Find course search links on a page
candidates = model.find_course_search_links(url, html_content)
```

## Training Data Collection

The model collects training data by:
1. Generating candidate URLs for each university domain
2. Visiting each URL and analyzing the content
3. Using heuristics to determine if it's a course search page
4. Storing the URL, HTML content, and classification

### Candidate URL Generation

The model generates URLs using patterns like:
- `https://{domain}/catalog`
- `https://{domain}/courses`
- `https://catalog.{domain}`
- `https://courses.{domain}`
- `https://{domain}/course-search`
- And many more...

## Model Performance

### Current Performance Metrics
- **Accuracy**: Target >80% on test set
- **Precision**: High precision for course search pages
- **Recall**: Good recall for finding course search functionality
- **F1-Score**: Balanced precision and recall

### Evaluation Criteria
- Correctly identifies course search pages
- Avoids false positives on general university pages
- Handles different university website structures
- Works across various course search interfaces

## Advanced Features

### 1. **Export Functionality Detection**
Strong indicator of course search pages:
- "Export all results"
- "Export as CSV"
- "Download results"

### 2. **Results Section Detection**
Indicates search functionality:
- "Results (X)"
- "Search results"
- "No results found"

### 3. **Advanced Filtering Detection**
Shows sophisticated search capabilities:
- "Sort by"
- "Filter by"
- "Departments"
- "Subjects"
- "Terms"

### 4. **Link Relationship Analysis**
Analyzes all links to find course search candidates:
- Scores links based on text and URL patterns
- Identifies navigation hierarchies
- Finds course-related navigation items

## Future Enhancements

### 1. **Graph Neural Network Integration**
```python
# Future implementation
import torch_geometric
from torch_geometric.nn import GCNConv

class CourseSearchGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 1)
```

### 2. **Transformer-based Text Analysis**
```python
# Future implementation
from transformers import AutoTokenizer, AutoModel

class CourseSearchTransformer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
```

### 3. **Multi-Modal Fusion**
```python
# Future implementation
class MultiModalCourseSearchModel:
    def __init__(self):
        self.text_model = CourseSearchTransformer()
        self.graph_model = CourseSearchGNN()
        self.fusion_layer = torch.nn.Linear(768 + 64, 1)
```

## Best Practices

### 1. **Feature Engineering**
- Focus on domain-specific features
- Combine multiple types of analysis
- Use interpretable features for debugging

### 2. **Training Data Quality**
- Collect diverse university websites
- Include both positive and negative examples
- Validate training data manually

### 3. **Model Evaluation**
- Use cross-validation
- Test on unseen universities
- Monitor for overfitting

### 4. **Deployment Considerations**
- Handle timeouts gracefully
- Implement retry logic
- Cache results when possible

## Comparison with Existing Models

### vs. Rule-based Approaches
- **Advantage**: Learns patterns automatically
- **Advantage**: Handles edge cases better
- **Advantage**: Improves with more data

### vs. Simple ML Models
- **Advantage**: Multi-modal analysis
- **Advantage**: Comprehensive feature engineering
- **Advantage**: Better interpretability

### vs. Deep Learning Only
- **Advantage**: Works with smaller datasets
- **Advantage**: More interpretable
- **Advantage**: Faster training and inference

## Conclusion

The Course Search Navigator Model provides a robust, multi-modal approach to finding course search pages from college catalog pages. It combines the strengths of traditional machine learning with comprehensive feature engineering, making it both effective and interpretable.

The model is designed to be:
- **Scalable**: Can handle thousands of universities
- **Robust**: Works across different website structures
- **Interpretable**: Provides detailed feature analysis
- **Extensible**: Easy to add new features or models

This approach should significantly improve the ability to automatically find course search functionality across diverse university websites. 