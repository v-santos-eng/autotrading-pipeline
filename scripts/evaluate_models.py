#!/usr/bin/env python3
"""
Advanced model evaluation and ensemble methods for walk-forward CV.

This script provides ensemble methods, model comparison, and advanced evaluation
techniques for the trained models across CV folds.
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Advanced model evaluation and ensemble framework."""
    
    def __init__(self, models_dir: str = "data/external/models"):
        self.models_dir = Path(models_dir)
        self.results_file = self.models_dir / "training_results.json"
        self.ensemble_results = {}
        
    def load_fold_models(self, fold_idx: int) -> Dict[str, Tuple[Any, Any, StandardScaler]]:
        """Load all trained models for a specific fold."""
        fold_dir = self.models_dir / f"fold_{fold_idx}"
        if not fold_dir.exists():
            return {}
        
        models = {}
        scaler_path = fold_dir / "scaler.pkl"
        
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        else:
            scaler = None
        
        # Load individual model files
        for model_file in fold_dir.glob("*_model.pkl"):
            model_name = model_file.stem.replace("_model", "")
            try:
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                models[model_name] = (model, scaler)
            except Exception as e:
                print(f"Warning: Could not load {model_file}: {e}")
        
        return models
    
    def create_ensemble_model(self, fold_idx: int, ensemble_type: str = "voting") -> Optional[VotingClassifier]:
        """Create ensemble model from individual fold models."""
        models = self.load_fold_models(fold_idx)
        if not models:
            return None
        
        estimators = []
        for model_name, (model, _) in models.items():
            if hasattr(model, 'predict_proba'):  # Only models with probability prediction
                estimators.append((model_name, model))
        
        if len(estimators) < 2:
            print(f"Warning: Not enough models for ensemble in fold {fold_idx}")
            return None
        
        if ensemble_type == "voting":
            ensemble = VotingClassifier(
                estimators=estimators, 
                voting='soft',  # Use probability-based voting
                n_jobs=-1
            )
        elif ensemble_type == "bagging":
            # Use the best individual model as base estimator
            best_model_name = max(models.keys(), key=lambda x: self.get_model_performance(fold_idx, x))
            base_estimator = models[best_model_name][0]
            ensemble = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        return ensemble
    
    def get_model_performance(self, fold_idx: int, model_name: str) -> float:
        """Get F1 score for a specific model and fold."""
        if not self.results_file.exists():
            return 0.0
        
        with open(self.results_file) as f:
            results = json.load(f)
        
        fold_key = f"fold_{fold_idx}"
        if fold_key in results and model_name in results[fold_key]:
            model_results = results[fold_key][model_name]
            if 'metrics' in model_results and 'f1' in model_results['metrics']:
                return model_results['metrics']['f1']
        
        return 0.0
    
    def evaluate_ensemble(self, fold_idx: int, X_test: np.ndarray, y_test: np.ndarray, 
                         ensemble_type: str = "voting") -> Dict[str, Any]:
        """Evaluate ensemble model performance."""
        ensemble = self.create_ensemble_model(fold_idx, ensemble_type)
        if ensemble is None:
            return {'error': 'Could not create ensemble model'}
        
        # For voting classifier, we need to refit on the training data
        # This is a simplified approach - in practice, you'd want to load training data
        # For now, we'll evaluate on test data only
        try:
            y_pred = ensemble.predict(X_test)
            y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'ensemble_type': ensemble_type
            }
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def plot_roc_curves(self, fold_idx: int, X_test: np.ndarray, y_test: np.ndarray, save_path: Optional[Path] = None):
        """Plot ROC curves for all models in a fold."""
        models = self.load_fold_models(fold_idx)
        if not models:
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, (model, _) in models.items():
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
                except Exception as e:
                    print(f"Warning: Could not plot ROC for {model_name}: {e}")
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - Fold {fold_idx}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curves(self, fold_idx: int, X_test: np.ndarray, y_test: np.ndarray, save_path: Optional[Path] = None):
        """Plot Precision-Recall curves for all models in a fold."""
        models = self.load_fold_models(fold_idx)
        if not models:
            return
        
        plt.figure(figsize=(10, 8))
        
        for model_name, (model, _) in models.items():
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                    
                    plt.plot(recall, precision, label=f'{model_name}', linewidth=2)
                except Exception as e:
                    print(f"Warning: Could not plot PR curve for {model_name}: {e}")
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - Fold {fold_idx}')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_ensemble_comparison(self, num_folds: int = 5) -> Dict[str, List[float]]:
        """Compare ensemble methods across multiple folds."""
        ensemble_comparison = {
            'voting_f1': [],
            'bagging_f1': [],
            'best_individual_f1': []
        }
        
        for fold_idx in range(num_folds):
            models = self.load_fold_models(fold_idx)
            if not models:
                continue
            
            # Get best individual model performance
            best_f1 = max(self.get_model_performance(fold_idx, model_name) for model_name in models.keys())
            ensemble_comparison['best_individual_f1'].append(best_f1)
            
            # Note: In a real implementation, you would load the actual test data
            # and evaluate ensemble performance. This is a simplified comparison.
            print(f"Fold {fold_idx}: Best individual F1 = {best_f1:.3f}")
        
        return ensemble_comparison
    
    def plot_model_stability(self, model_name: str, save_path: Optional[Path] = None) -> Dict[str, List[float]]:
        """Plot model performance stability across all folds."""
        if not self.results_file.exists():
            return {}
        
        with open(self.results_file) as f:
            results = json.load(f)
        
        fold_performance = {'f1': [], 'accuracy': [], 'precision': [], 'recall': []}
        fold_indices = []
        
        for fold_key, fold_results in results.items():
            if model_name in fold_results and 'metrics' in fold_results[model_name]:
                metrics = fold_results[model_name]['metrics']
                fold_idx = int(fold_key.replace('fold_', ''))
                fold_indices.append(fold_idx)
                
                fold_performance['f1'].append(metrics.get('f1', 0))
                fold_performance['accuracy'].append(metrics.get('accuracy', 0))
                fold_performance['precision'].append(metrics.get('precision', 0))
                fold_performance['recall'].append(metrics.get('recall', 0))
        
        if not fold_performance['f1']:
            print(f"No performance data found for model: {model_name}")
            return {}
        
        # Create stability plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics = ['f1', 'accuracy', 'precision', 'recall']
        titles = ['F1 Score', 'Accuracy', 'Precision', 'Recall']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            values = fold_performance[metric]
            
            ax.plot(fold_indices, values, 'o-', linewidth=2, markersize=6, alpha=0.8)
            ax.axhline(y=np.mean(values), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(values):.3f}')
            ax.fill_between(fold_indices, np.mean(values) - np.std(values), np.mean(values) + np.std(values), 
                           alpha=0.2, color='red', label=f'±1 STD: {np.std(values):.3f}')
            
            ax.set_xlabel('Fold Index')
            ax.set_ylabel(title)
            ax.set_title(f'{model_name} - {title} Stability')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.suptitle(f'Model Performance Stability: {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fold_performance
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not self.results_file.exists():
            return {'error': 'No training results found'}
        
        with open(self.results_file) as f:
            results = json.load(f)
        
        report = {
            'summary': {},
            'model_comparison': {},
            'stability_analysis': {},
            'ensemble_comparison': {}
        }
        
        # Model performance summary
        model_performance = {}
        for fold_key, fold_results in results.items():
            for model_name, model_result in fold_results.items():
                if 'metrics' in model_result:
                    if model_name not in model_performance:
                        model_performance[model_name] = {'f1': [], 'accuracy': [], 'auc': []}
                    
                    metrics = model_result['metrics']
                    model_performance[model_name]['f1'].append(metrics.get('f1', 0))
                    model_performance[model_name]['accuracy'].append(metrics.get('accuracy', 0))
                    model_performance[model_name]['auc'].append(metrics.get('roc_auc', 0))
        
        # Calculate summary statistics
        for model_name, performance in model_performance.items():
            report['summary'][model_name] = {
                'avg_f1': np.mean(performance['f1']),
                'std_f1': np.std(performance['f1']),
                'avg_accuracy': np.mean(performance['accuracy']),
                'std_accuracy': np.std(performance['accuracy']),
                'avg_auc': np.mean([x for x in performance['auc'] if x is not None]) if performance['auc'] else 0,
                'folds_trained': len(performance['f1'])
            }
        
        # Find best performing model
        if report['summary']:
            best_model = max(report['summary'].items(), key=lambda x: x[1]['avg_f1'])
            report['best_model'] = {
                'name': best_model[0],
                'avg_f1': best_model[1]['avg_f1'],
                'std_f1': best_model[1]['std_f1']
            }
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Advanced model evaluation and ensemble methods")
    parser.add_argument("--fold", type=int, help="Specific fold to evaluate")
    parser.add_argument("--model", type=str, help="Specific model to analyze")
    parser.add_argument("--ensemble-type", choices=["voting", "bagging"], default="voting", help="Ensemble method")
    parser.add_argument("--plot-roc", action="store_true", help="Plot ROC curves")
    parser.add_argument("--plot-pr", action="store_true", help="Plot Precision-Recall curves")
    parser.add_argument("--plot-stability", action="store_true", help="Plot model stability")
    parser.add_argument("--compare-ensemble", action="store_true", help="Compare ensemble methods")
    parser.add_argument("--comprehensive-report", action="store_true", help="Generate comprehensive report")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    
    if args.comprehensive_report:
        report = evaluator.generate_comprehensive_report()
        print("\n📊 Comprehensive Model Evaluation Report")
        print("=" * 60)
        
        if 'error' in report:
            print(f"❌ Error: {report['error']}")
            return
        
        # Print summary
        print("\n🎯 Model Performance Summary")
        print("-" * 40)
        print(f"{'Model':<15} {'Avg F1':<10} {'Std F1':<10} {'Avg Acc':<10} {'Folds':<6}")
        print("-" * 55)
        
        for model_name, stats in report['summary'].items():
            print(f"{model_name:<15} {stats['avg_f1']:<10.3f} {stats['std_f1']:<10.3f} {stats['avg_accuracy']:<10.3f} {stats['folds_trained']:<6}")
        
        if 'best_model' in report:
            best = report['best_model']
            print(f"\n🏆 Best Model: {best['name']} (F1: {best['avg_f1']:.3f} ± {best['std_f1']:.3f})")
        
        # Save report
        report_path = Path("data/external/models/evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✅ Comprehensive report saved to {report_path}")
        
    elif args.model and args.plot_stability:
        # Plot model stability
        save_path = Path(f"data/external/models/{args.model}_stability.png")
        performance = evaluator.plot_model_stability(args.model, save_path)
        
        if performance:
            print(f"\n📈 Model stability plot saved to {save_path}")
            print(f"F1 Score: {np.mean(performance['f1']):.3f} ± {np.std(performance['f1']):.3f}")
            print(f"Accuracy: {np.mean(performance['accuracy']):.3f} ± {np.std(performance['accuracy']):.3f}")
    
    elif args.compare_ensemble:
        # Compare ensemble methods
        comparison = evaluator.generate_ensemble_comparison()
        print("\n🔍 Ensemble Method Comparison")
        print("-" * 40)
        print("(Note: This is a simplified comparison)")
        print(f"Best individual models average F1: {np.mean(comparison['best_individual_f1']):.3f}")
    
    else:
        print("\n🎯 Model Evaluation Framework")
        print("Use --comprehensive-report for full analysis")
        print("Use --model <name> --plot-stability for stability analysis")
        print("Use --compare-ensemble for ensemble comparison")

if __name__ == "__main__":
    main()