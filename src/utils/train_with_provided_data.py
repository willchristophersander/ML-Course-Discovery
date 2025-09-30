#!/usr/bin/env python3
"""
Train Course Search Navigator Model with Provided Data

This script uses the analyzed course search and catalog data to train
the course search navigator model.
"""

import json
import logging
import os
import sys

# Ensure the src/ package hierarchy is importable when executed directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.course_search_navigator_model import CourseSearchNavigatorModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CourseSearchModelTrainer:
    """
    Trainer for the course search navigator model using provided data.
    """
    
    def __init__(self):
        self.model = CourseSearchNavigatorModel()
        self.training_data = []
        self.test_data = []
        
    def load_analyzed_data(self, filename='course_search_training_data.json'):
        """
        Load the analyzed training data.
        """
        try:
            with open(filename, 'r') as f:
                self.training_data = json.load(f)
            logger.info(f"Loaded {len(self.training_data)} training examples")
            return True
        except FileNotFoundError:
            logger.error(f"Training data file {filename} not found!")
            return False
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return False
    
    def prepare_training_data(self):
        """
        Prepare the training data for the model.
        """
        logger.info("Preparing training data...")
        
        # Extract features for all training examples
        features_list = []
        labels = []
        
        for item in self.training_data:
            try:
                features, _, _ = self.model.extract_comprehensive_features(
                    item['url'], item['html_content']
                )
                
                if features:
                    features_list.append(features)
                    # Use is_course_search as the label (1 for course search, 0 for catalog)
                    labels.append(1 if item['is_course_search'] else 0)
                    
            except Exception as e:
                logger.warning(f"Error processing {item['url']}: {e}")
        
        if len(features_list) < 5:
            logger.error("Insufficient training data!")
            return False, None, None
        
        # Convert to numpy arrays
        feature_names = list(features_list[0].keys())
        X = np.array([[f.get(name, 0) for name in feature_names] for f in features_list])
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} examples")
        logger.info(f"Test set: {len(X_test)} examples")
        logger.info(f"Feature names: {feature_names}")
        
        return True, (X_train, X_test, y_train, y_test), feature_names
    
    def train_model(self, X_train, y_train, feature_names):
        """
        Train the course search navigator model.
        """
        logger.info("Training course search navigator model...")
        
        # Scale features
        X_train_scaled = self.model.scaler.fit_transform(X_train)
        
        # Train the model
        success = self.model.train_model([
            {
                'url': item['url'],
                'html_content': item['html_content'],
                'is_course_search': bool(label)
            }
            for item, label in zip(self.training_data, y_train)
        ])
        
        if success:
            logger.info("Model training completed successfully!")
            return True
        else:
            logger.error("Model training failed!")
            return False
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model.
        """
        logger.info("Evaluating model performance...")
        
        # Scale test features
        X_test_scaled = self.model.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.model.predict(X_test_scaled)
        y_pred_proba = self.model.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info("Model Performance:")
        logger.info(f"Accuracy: {accuracy:.3f}")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"F1-Score: {f1:.3f}")
        
        # Detailed classification report
        report = classification_report(y_test, y_pred, target_names=['Catalog', 'Course Search'])
        logger.info(f"\nClassification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def test_model_on_examples(self):
        """
        Test the model on specific examples.
        """
        logger.info("Testing model on specific examples...")
        
        test_cases = [
            {
                'url': 'https://globalsearch.cuny.edu/',
                'description': 'CUNY Global Class Search',
                'expected': True
            },
            {
                'url': 'https://classes.berkeley.edu/',
                'description': 'UC Berkeley Class Schedule',
                'expected': True
            },
            {
                'url': 'https://catalog.unc.edu/',
                'description': 'UNC Chapel Hill Catalog',
                'expected': False
            },
            {
                'url': 'https://catalog.utexas.edu/',
                'description': 'UT Austin Catalog',
                'expected': False
            }
        ]
        
        correct_predictions = 0
        total_predictions = 0
        
        for test_case in test_cases:
            url = test_case['url']
            description = test_case['description']
            expected = test_case['expected']
            
            # Find the corresponding training data
            for item in self.training_data:
                if item['url'] == url:
                    try:
                        is_course_search, confidence, details = self.model.predict_course_search(
                            url, item['html_content']
                        )
                        
                        correct = (is_course_search == expected)
                        if correct:
                            correct_predictions += 1
                        total_predictions += 1
                        
                        status = "" if correct else ""
                        logger.info(f"{status} {description}")
                        logger.info(f"  URL: {url}")
                        logger.info(f"  Expected: {'Course Search' if expected else 'Catalog'}")
                        logger.info(f"  Predicted: {'Course Search' if is_course_search else 'Catalog'}")
                        logger.info(f"  Confidence: {confidence:.3f}")
                        
                        if 'features' in details:
                            features = details['features']
                            logger.info(f"  Search patterns: {features.get('search_pattern_matches', 0)}")
                            logger.info(f"  URL patterns: {features.get('url_pattern_matches', 0)}")
                            logger.info(f"  Export functionality: {features.get('has_export_functionality', False)}")
                            logger.info(f"  Results section: {features.get('has_results_section', False)}")
                            logger.info(f"  Advanced filtering: {features.get('has_advanced_filtering', False)}")
                        
                        break
                        
                    except Exception as e:
                        logger.error(f"Error testing {url}: {e}")
        
        if total_predictions > 0:
            accuracy = correct_predictions / total_predictions
            logger.info(f"\nTest Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
            return accuracy
        else:
            logger.error("No test predictions made!")
            return 0.0
    
    def save_model_info(self, metrics, feature_names):
        """
        Save model information and metrics.
        """
        model_info = {
            'training_samples': len(self.training_data),
            'feature_names': feature_names,
            'metrics': metrics,
            'model_file': self.model.model_file
        }
        
        with open('course_search_model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("Model information saved to course_search_model_info.json")

def main():
    """Main training script."""
    
    print(" TRAINING COURSE SEARCH NAVIGATOR MODEL")
    print("=" * 60)
    
    trainer = CourseSearchModelTrainer()
    
    # Load analyzed data
    print("\n Loading analyzed data...")
    if not trainer.load_analyzed_data():
        print(" Failed to load training data!")
        return
    
    # Prepare training data
    print("\n Preparing training data...")
    success, data, feature_names = trainer.prepare_training_data()
    
    if not success:
        print(" Failed to prepare training data!")
        return
    
    X_train, X_test, y_train, y_test = data
    
    # Train the model
    print("\n Training the model...")
    if trainer.train_model(X_train, y_train, feature_names):
        print(" Model training completed!")
        
        # Evaluate the model
        print("\n Evaluating model performance...")
        metrics = trainer.evaluate_model(X_test, y_test)
        
        # Test on specific examples
        print("\n Testing on specific examples...")
        test_accuracy = trainer.test_model_on_examples()
        
        # Save model information
        trainer.save_model_info(metrics, feature_names)
        
        print(f"\n TRAINING COMPLETE!")
        print("=" * 40)
        print(f" Model Accuracy: {metrics['accuracy']:.3f}")
        print(f" Precision: {metrics['precision']:.3f}")
        print(f" Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f" Test Accuracy: {test_accuracy:.3f}")
        
        if metrics['accuracy'] >= 0.8:
            print(" Excellent! Model performs very well!")
        elif metrics['accuracy'] >= 0.6:
            print(" Good! Model shows promise!")
        else:
            print("  Model needs more training data or tuning.")
        
        print(f"\n Model saved to: {trainer.model.model_file}")
        print(f" Model info saved to: course_search_model_info.json")
        
    else:
        print(" Model training failed!")

if __name__ == "__main__":
    main() 
