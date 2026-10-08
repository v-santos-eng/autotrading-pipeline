#!/usr/bin/env python3
"""
Comprehensive model training and evaluation framework for walk-forward CV splits.

This script trains multiple models on each CV fold, evaluates performance,
and provides detailed analysis including feature importance and model comparison.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Comprehensive model training and evaluation framework."""
    
    def __init__(self, data_dir: str = "data", models_dir: str = "data/external/models"):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Define models and their hyperparameter grids
        self.models = {
            'rf': {
                'name': 'Random Forest',
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'gb': {
                'name': 'Gradient Boosting',
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.1, 0.01],
                    'max_depth': [3, 5],
                    'subsample': [0.8, 1.0]
                }
            },
            'lr': {
                'name': 'Logistic Regression',
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'svm': {
                'name': 'Support Vector Machine',
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        
        self.results = {}
        self.feature_importance = {}
        
    def load_fold_data(self, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Load train and test data for a specific fold."""
        splits_dir = self.data_dir / "splits"
        
        train_file = splits_dir / f"train_fold_{fold_idx}.parquet"
        test_file = splits_dir / f"test_fold_{fold_idx}.parquet"
        
        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(f"Fold {fold_idx} data not found")
        
        train_df = pd.read_parquet(train_file)
        test_df = pd.read_parquet(test_file)
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col not in ['label', 'timestamp', 'symbol']]
        
        X_train = train_df[feature_cols]
        y_train = train_df['label']
        X_test = test_df[feature_cols]
        y_test = test_df['label']
        
        return X_train, X_test, y_train, y_test
    
    def prepare_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Prepare features by handling missing values and scaling."""
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use train median for test
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, scaler
    
    def train_model_with_cv(self, model_config: Dict, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train a model with hyperparameter tuning using GridSearchCV."""
        model = model_config['model']
        params = model_config['params']
        
        # Use a smaller CV for efficiency (3-fold instead of 5-fold)
        grid_search = GridSearchCV(
            model, params, cv=3, scoring='f1', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
        """Comprehensive model evaluation with multiple metrics."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['roc_auc'] = None
        
        # Detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def extract_feature_importance(self, model: Any, feature_names: List[str], model_type: str) -> Dict[str, float]:
        """Extract feature importance from trained models."""
        importance = {}
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            for name, imp in zip(feature_names, model.feature_importances_):
                importance[name] = float(imp)
        elif hasattr(model, 'coef_'):
            # Linear models
            for name, coef in zip(feature_names, model.coef_[0]):
                importance[name] = float(abs(coef))
        else:
            # Other models - use permutation importance (simplified)
            importance = {name: 0.0 for name in feature_names}
        
        return importance
    
    def train_fold_models(self, fold_idx: int, skip_existing: bool = True) -> Dict[str, Any]:
        """Train all models for a specific fold."""
        print(f"\n🔄 Training models for fold {fold_idx}...")
        
        fold_results = {}
        fold_dir = self.models_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)
        
        # Check if models already exist
        if skip_existing and (fold_dir / "results.json").exists():
            print(f"  ✅ Fold {fold_idx} already trained, skipping...")
            with open(fold_dir / "results.json") as f:
                return json.load(f)
        
        try:
            # Load fold data
            X_train, X_test, y_train, y_test = self.load_fold_data(fold_idx)
            
            # Prepare features
            X_train_scaled, X_test_scaled, scaler = self.prepare_features(X_train, X_test)
            
            # Save scaler
            with open(fold_dir / "scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            
            feature_names = X_train.columns.tolist()
            
            # Train each model
            for model_key, model_config in self.models.items():
                print(f"  Training {model_config['name']}...")
                
                try:
                    # Train model with hyperparameter tuning
                    trained_model = self.train_model_with_cv(model_config, X_train_scaled, y_train)
                    
                    # Evaluate model
                    metrics = self.evaluate_model(trained_model, X_test_scaled, y_test, model_config['name'])
                    
                    # Extract feature importance
                    importance = self.extract_feature_importance(trained_model, feature_names, model_key)
                    
                    # Save model and results
                    model_path = fold_dir / f"{model_key}_model.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(trained_model, f)
                    
                    fold_results[model_key] = {
                        'model_name': model_config['name'],
                        'metrics': metrics,
                        'feature_importance': importance,
                        'best_params': trained_model.get_params() if hasattr(trained_model, 'get_params') else {}
                    }
                    
                    print(f"    ✅ {model_config['name']}: F1={metrics['f1']:.3f}, AUC={metrics.get('roc_auc', 'N/A')}")
                    
                except Exception as e:
                    print(f"    ❌ {model_config['name']}: {str(e)}")
                    fold_results[model_key] = {
                        'model_name': model_config['name'],
                        'error': str(e)
                    }
            
            # Save fold results
            with open(fold_dir / "results.json", "w") as f:
                json.dump(fold_results, f, indent=2, default=str)
            
            return fold_results
            
        except Exception as e:
            print(f"  ❌ Failed to train fold {fold_idx}: {str(e)}")
            return {'error': str(e)}
    
    def train_all_folds(self, start_fold: int = 0, end_fold: int = None, skip_existing: bool = True):
        """Train models on all CV folds."""
        # Load manifest to get total number of folds
        manifest_path = self.data_dir / "splits" / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("CV splits manifest not found. Run export_walk_forward_splits.py first.")
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        total_folds = len(manifest)
        if end_fold is None:
            end_fold = total_folds - 1
        
        print(f"\n🎯 Training models on folds {start_fold} to {end_fold} (total: {total_folds})")
        
        all_results = {}
        for fold_idx in range(start_fold, end_fold + 1):
            try:
                fold_results = self.train_fold_models(fold_idx, skip_existing)
                all_results[f"fold_{fold_idx}"] = fold_results
            except Exception as e:
                print(f"❌ Failed to process fold {fold_idx}: {str(e)}")
                all_results[f"fold_{fold_idx}"] = {'error': str(e)}
        
        # Save comprehensive results
        results_path = self.models_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n✅ Training complete! Results saved to {results_path}")
        return all_results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report across all folds and models."""
        results_path = self.models_dir / "training_results.json"
        if not results_path.exists():
            print("❌ No training results found. Run training first.")
            return
        
        with open(results_path) as f:
            all_results = json.load(f)
        
        print("\n📊 Model Performance Report")
        print("=" * 60)
        
        # Aggregate metrics across folds
        model_metrics = {}
        for fold_key, fold_results in all_results.items():
            if 'error' in fold_results:
                continue
                
            for model_key, model_result in fold_results.items():
                if 'error' in model_result or 'metrics' not in model_result:
                    continue
                
                if model_key not in model_metrics:
                    model_metrics[model_key] = {
                        'name': model_result['model_name'],
                        'accuracies': [],
                        'precisions': [],
                        'recalls': [],
                        'f1_scores': [],
                        'auc_scores': []
                    }
                
                metrics = model_result['metrics']
                model_metrics[model_key]['accuracies'].append(metrics['accuracy'])
                model_metrics[model_key]['precisions'].append(metrics['precision'])
                model_metrics[model_key]['recalls'].append(metrics['recall'])
                model_metrics[model_key]['f1_scores'].append(metrics['f1'])
                if metrics.get('roc_auc') is not None:
                    model_metrics[model_key]['auc_scores'].append(metrics['roc_auc'])
        
        # Calculate summary statistics
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
        print("-" * 70)
        
        for model_key, metrics in model_metrics.items():
            acc_mean = np.mean(metrics['accuracies'])
            acc_std = np.std(metrics['accuracies'])
            prec_mean = np.mean(metrics['precisions'])
            prec_std = np.std(metrics['precisions'])
            rec_mean = np.mean(metrics['recalls'])
            rec_std = np.std(metrics['recalls'])
            f1_mean = np.mean(metrics['f1_scores'])
            f1_std = np.std(metrics['f1_scores'])
            
            auc_mean = np.mean(metrics['auc_scores']) if metrics['auc_scores'] else 0
            auc_str = f"{auc_mean:.3f}" if metrics['auc_scores'] else "N/A"
            
            print(f"{metrics['name']:<20} {acc_mean:.3f}±{acc_std:.3f} {prec_mean:.3f}±{prec_std:.3f} {rec_mean:.3f}±{rec_std:.3f} {f1_mean:.3f}±{f1_std:.3f} {auc_str:<10}")
        
        # Find best model
        if model_metrics:
            best_model = max(model_metrics.items(), key=lambda x: np.mean(x[1]['f1_scores']))
            print(f"\n🏆 Best Model: {best_model[1]['name']} (Avg F1: {np.mean(best_model[1]['f1_scores']):.3f})")
        else:
            print("\n⚠️  No successful model training results found")
        
        return model_metrics
    
    def plot_feature_importance(self, top_n: int = 15):
        """Plot aggregated feature importance across all folds and models."""
        results_path = self.models_dir / "training_results.json"
        if not results_path.exists():
            print("❌ No training results found. Run training first.")
            return
        
        with open(results_path) as f:
            all_results = json.load(f)
        
        # Aggregate feature importance
        feature_importance_agg = {}
        model_counts = {}
        
        for fold_key, fold_results in all_results.items():
            if 'error' in fold_results:
                continue
                
            for model_key, model_result in fold_results.items():
                if 'error' in model_result or 'feature_importance' not in model_result:
                    continue
                
                model_name = model_result['model_name']
                if model_name not in feature_importance_agg:
                    feature_importance_agg[model_name] = {}
                    model_counts[model_name] = 0
                
                model_counts[model_name] += 1
                for feature, importance in model_result['feature_importance'].items():
                    if feature not in feature_importance_agg[model_name]:
                        feature_importance_agg[model_name][feature] = []
                    feature_importance_agg[model_name][feature].append(importance)
        
        # Calculate average importance
        for model_name in feature_importance_agg:
            for feature in feature_importance_agg[model_name]:
                feature_importance_agg[model_name][feature] = np.mean(feature_importance_agg[model_name][feature])
        
        # Create plots
        n_models = len(feature_importance_agg)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models), squeeze=False)
        
        for idx, (model_name, importance_dict) in enumerate(feature_importance_agg.items()):
            ax = axes[idx, 0]
            
            # Sort features by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            features, importances = zip(*sorted_features)
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{model_name} - Top {top_n} Features')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(importances):
                ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plot_path = self.models_dir / "feature_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📈 Feature importance plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Train models on walk-forward CV splits")
    parser.add_argument("--start-fold", type=int, default=0, help="Start fold index")
    parser.add_argument("--end-fold", type=int, help="End fold index (default: all)")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip existing trained models")
    parser.add_argument("--report-only", action="store_true", help="Generate report without training")
    parser.add_argument("--plot-importance", action="store_true", help="Plot feature importance")
    
    args = parser.parse_args()
    
    trainer = ModelTrainer()
    
    if args.report_only:
        trainer.generate_performance_report()
        if args.plot_importance:
            trainer.plot_feature_importance()
    else:
        # Train models
        all_results = trainer.train_all_folds(args.start_fold, args.end_fold, args.skip_existing)
        
        # Generate report
        trainer.generate_performance_report()
        
        if args.plot_importance:
            trainer.plot_feature_importance()
    
    print("\n✅ Model training and evaluation complete!")

if __name__ == "__main__":
    main()