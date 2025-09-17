import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TicketClassificationSystem:
    def __init__(self, data_path="C:\\Users\\ibrahim.fadhili\\OneDrive - Agile Business Solutions\\Desktop\\Customer Support\\data\\processed\\processed_tickets.csv"):
        """
        Initialize the classification system
        """
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.results = {}
        
    def load_data(self):
        """
        Load and prepare the dataset
        """
        print("Loading data...")
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            print("\nDataset info:")
            print(self.data.info())
            print("\nClass distributions:")
            for col in ['type', 'queue', 'priority']:
                if col in self.data.columns:
                    print(f"\n{col.upper()}:")
                    print(self.data[col].value_counts())
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found!")
            return False
        return True
    
    def preprocess_text(self, text_column='combined_text'):
        """
        Combine subject and body, clean text
        """
        print("Preprocessing text...")
        # Combine subject and body
        self.data['combined_text'] = (
            self.data['subject'].fillna('') + ' ' + 
            self.data['body'].fillna('')
        ).str.strip()
        
        # Basic text cleaning
        self.data['combined_text'] = self.data['combined_text'].str.lower()
        
        print(f"Text preprocessing complete. Average text length: {self.data['combined_text'].str.len().mean():.1f} chars")
    
    def prepare_features_and_targets(self):
        """
        Prepare X (features) and y (targets) for each classification task
        """
        print("Preparing features and targets...")
        
        # Features (text)
        X = self.data['combined_text']
        
        # Targets for each classification task
        targets = {}
        
        if 'type' in self.data.columns:
            targets['type'] = self.data['type']
            
        if 'queue' in self.data.columns:
            targets['queue'] = self.data['queue']
            
        if 'priority' in self.data.columns:
            targets['priority'] = self.data['priority']
        
        return X, targets
    
    def get_model_configs(self):
        """
        Define model configurations
        """
        return {
            'logistic': {
                'name': 'Logistic Regression',
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'tfidf__max_features': [1000, 5000, 10000],
                    'tfidf__ngram_range': [(1, 1), (1, 2)]
                }
            },
            'random_forest': {
                'name': 'Random Forest',
                'model': RandomForestClassifier(random_state=42, n_estimators=100),
                'params': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'tfidf__max_features': [1000, 5000]
                }
            },
            'svm': {
                'name': 'Support Vector Machine',
                'model': SVC(random_state=42),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['linear', 'rbf'],
                    'tfidf__max_features': [1000, 5000]
                }
            }
        }
    
    def train_model(self, task_name, X, y, model_type='logistic', use_smote=True, test_size=0.2):
        """
        Train a single classification model
        """
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model for {task_name.upper()} classification")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Get model configuration
        model_configs = self.get_model_configs()
        config = model_configs[model_type]
        
        # Create pipeline
        if use_smote and len(y.unique()) > 1:
            # Use SMOTE for handling class imbalance
            pipeline = ImbPipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
                ('smote', SMOTE(random_state=42)),
                ('classifier', config['model'])
            ])
        else:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
                ('classifier', config['model'])
            ])
        
        # Grid search with cross-validation
        print("Performing grid search...")
        grid_search = GridSearchCV(
            pipeline,
            config['params'],
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        # Train model
        start_time = datetime.now()
        grid_search.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_weighted')
        
        # Store results
        results = {
            'model': best_model,
            'accuracy': accuracy,
            'f1_score': f1_weighted,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_,
            'training_time': training_time,
            'y_test': y_test,
            'y_pred': y_pred,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Store in class attributes
        self.models[f"{task_name}_{model_type}"] = best_model
        self.results[f"{task_name}_{model_type}"] = results
        
        # Print results
        self.print_model_results(task_name, model_type, results)
        
        return results
    
    def print_model_results(self, task_name, model_type, results):
        """
        Print detailed model results
        """
        print(f"\n{'-'*40}")
        print(f"RESULTS: {task_name.upper()} - {model_type.upper()}")
        print(f"{'-'*40}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1-Score (weighted): {results['f1_score']:.4f}")
        print(f"CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        
        print("\nBest Parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\nClassification Report:")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        print(report_df.round(3))
        
        print(f"\nConfusion Matrix:")
        print(results['confusion_matrix'])
    
    def train_all_models(self, tasks=['type', 'queue', 'priority'], models=['logistic', 'random_forest']):
        """
        Train all models for all tasks
        """
        if not self.load_data():
            return
        
        self.preprocess_text()
        X, targets = self.prepare_features_and_targets()
        
        print(f"\n🚀 Starting training for {len(tasks)} tasks with {len(models)} algorithms each...")
        print(f"Tasks: {tasks}")
        print(f"Models: {models}")
        
        for task in tasks:
            if task not in targets:
                print(f"Warning: Task '{task}' not found in data. Skipping...")
                continue
                
            y = targets[task]
            print(f"\n📊 Task: {task.upper()} - Classes: {list(y.unique())}")
            
            for model_type in models:
                try:
                    self.train_model(task, X, y, model_type)
                except Exception as e:
                    print(f"Error training {model_type} for {task}: {str(e)}")
        
        # Generate summary
        self.generate_summary()
    
    def generate_summary(self):
        """
        Generate a summary comparison of all models
        """
        print(f"\n{'='*80}")
        print("MODEL PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        summary_data = []
        
        for key, results in self.results.items():
            task, model = key.split('_', 1)
            summary_data.append({
                'Task': task.upper(),
                'Model': model.replace('_', ' ').title(),
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'CV Score': f"{results['cv_mean']:.4f} ± {results['cv_std']:.4f}",
                'Training Time (s)': f"{results['training_time']:.2f}"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))
            
            # Find best models for each task
            print(f"\n{'='*50}")
            print("BEST MODELS BY TASK")
            print(f"{'='*50}")
            
            for task in summary_df['Task'].unique():
                task_results = summary_df[summary_df['Task'] == task]
                best_model = task_results.loc[task_results['F1-Score'].idxmax()]
                print(f"{task}: {best_model['Model']} (F1: {best_model['F1-Score']})")
    
    def save_models(self, save_dir="models/"):
        """
        Save trained models and vectorizers
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving {len(self.models)} models to {save_dir}...")
        
        for name, model in self.models.items():
            filepath = os.path.join(save_dir, f"{name}.joblib")
            joblib.dump(model, filepath)
            print(f"Saved: {filepath}")
        
        # Save results summary
        results_path = os.path.join(save_dir, "results_summary.txt")
        with open(results_path, 'w') as f:
            f.write("MODEL TRAINING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            for key, results in self.results.items():
                f.write(f"{key.upper()}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"F1-Score: {results['f1_score']:.4f}\n")
                f.write(f"CV Score: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}\n")
                f.write(f"Training Time: {results['training_time']:.2f}s\n\n")
        
        print(f"Results summary saved to: {results_path}")
    
    def predict_ticket(self, subject, body, model_names=None):
        """
        Predict classification for a new ticket
        """
        if not self.models:
            print("No models trained yet. Please run train_all_models() first.")
            return None
        
        # Prepare text
        combined_text = f"{subject} {body}".strip().lower()
        
        predictions = {}
        
        # Use best models for each task if not specified
        if model_names is None:
            model_names = {}
            for task in ['type', 'queue', 'priority']:
                # Find best model for this task based on F1 score
                task_models = [key for key in self.results.keys() if key.startswith(task)]
                if task_models:
                    best_model = max(task_models, 
                                   key=lambda x: self.results[x]['f1_score'])
                    model_names[task] = best_model
        
        for task, model_name in model_names.items():
            if model_name in self.models:
                model = self.models[model_name]
                pred = model.predict([combined_text])[0]
                
                # Get prediction probability if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba([combined_text])[0]
                    confidence = max(proba)
                else:
                    confidence = None
                
                predictions[task] = {
                    'prediction': pred,
                    'confidence': confidence
                }
        
        return predictions

# Example usage and main execution
if __name__ == "__main__":
    # Initialize the classification system
    classifier = TicketClassificationSystem("data/processed/processed_tickets.csv")
    
    # Train all models
    classifier.train_all_models(
        tasks=['type', 'queue', 'priority'],
        models=['logistic', 'random_forest']  # Start with these two
    )
    
    # Save trained models
    classifier.save_models()
    
    # Example prediction
    print(f"\n{'='*60}")
    print("EXAMPLE PREDICTION")
    print(f"{'='*60}")
    
    sample_subject = "System outage affecting all users"
    sample_body = "We are experiencing a critical system outage that is preventing all users from accessing our services. This is blocking business operations."
    
    predictions = classifier.predict_ticket(sample_subject, sample_body)
    
    if predictions:
        print(f"Subject: {sample_subject}")
        print(f"Body: {sample_body[:100]}...")
        print("\nPredictions:")
        for task, pred_info in predictions.items():
            conf_str = f" (confidence: {pred_info['confidence']:.3f})" if pred_info['confidence'] else ""
            print(f"  {task.upper()}: {pred_info['prediction']}{conf_str}")
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print("Models saved to 'models/' directory")
    print("Check results_summary.txt for detailed metrics")
    print("\nNext steps:")
    print("1. Review model performance metrics")
    print("2. Try different algorithms (add 'svm' to models list)")
    print("3. Experiment with BERT-based models for better accuracy")
    print("4. Collect more training data for improved performance")