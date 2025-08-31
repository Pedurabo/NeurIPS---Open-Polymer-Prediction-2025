"""
Pattern Evaluation for NeurIPS Open Polymer Prediction 2025
Cluster 5: Analyze prediction patterns, cross-validation, interpretability, and evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings
import json
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolymerModelEvaluator:
    """Comprehensive model evaluator for polymer property prediction"""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model evaluator"""
        self.random_state = random_state
        np.random.seed(random_state)
        self.target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.evaluation_results = {}
        self.cross_validation_results = {}
        self.interpretability_results = {}
        
    def perform_cross_validation(self, models: Dict[str, Any],
                                X: pd.DataFrame, y: pd.DataFrame,
                                cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation for all models"""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        cv_results = {}
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name, model_info in models.items():
            if model_info['status'] != 'success' or model_info['model'] is None:
                cv_results[model_name] = {'status': 'skipped', 'reason': 'Model not available'}
                continue
            
            try:
                logger.info(f"Cross-validating {model_name}...")
                model = model_info['model']
                
                # Initialize CV metrics
                cv_metrics = {'fold_scores': [], 'mean_scores': {}, 'std_scores': {}, 'status': 'success'}
                
                # Perform CV
                fold_scores = []
                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    X_train_fold = X.iloc[train_idx]
                    y_train_fold = y.iloc[train_idx]
                    X_val_fold = X.iloc[val_idx]
                    y_val_fold = y.iloc[val_idx]
                    
                    # Train model on this fold
                    if len(self.target_columns) > 1:
                        model.fit(X_train_fold, y_train_fold)
                        y_pred_fold = model.predict(X_val_fold)
                    else:
                        target_col = self.target_columns[0]
                        model.fit(X_train_fold, y_train_fold[target_col])
                        y_pred_fold = model.predict(X_val_fold).reshape(-1, 1)
                    
                    # Calculate metrics for this fold
                    fold_metrics = {}
                    for i, target in enumerate(self.target_columns):
                        if len(self.target_columns) > 1:
                            target_pred = y_pred_fold[:, i]
                            target_true = y_val_fold[target]
                        else:
                            target_pred = y_pred_fold.flatten()
                            target_true = y_val_fold[target]
                        
                        mae = mean_absolute_error(target_true, target_pred)
                        mse = mean_squared_error(target_true, target_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(target_true, target_pred)
                        
                        fold_metrics[target] = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
                    
                    # Calculate weighted MAE for this fold
                    weighted_mae = self._calculate_weighted_mae(y_val_fold, y_pred_fold)
                    fold_metrics['weighted_mae'] = weighted_mae
                    fold_scores.append(fold_metrics)
                
                # Calculate mean and std across folds
                cv_metrics['fold_scores'] = fold_scores
                
                # Aggregate metrics across targets
                for target in self.target_columns + ['weighted_mae']:
                    target_scores = [fold[target] for fold in fold_scores if target in fold]
                    if target_scores:
                        if isinstance(target_scores[0], dict):
                            # For target-specific metrics
                            metric_names = ['mae', 'mse', 'rmse', 'r2']
                            for metric in metric_names:
                                values = [score[metric] for score in target_scores]
                                cv_metrics['mean_scores'][f'{target}_{metric}'] = np.mean(values)
                                cv_metrics['std_scores'][f'{target}_{metric}'] = np.std(values)
                        else:
                            # For weighted MAE
                            cv_metrics['mean_scores'][target] = np.mean(target_scores)
                            cv_metrics['std_scores'][target] = np.std(target_scores)
                
                cv_results[model_name] = cv_metrics
                logger.info(f"✅ {model_name} CV completed successfully")
                
            except Exception as e:
                logger.error(f"❌ CV failed for {model_name}: {e}")
                cv_results[model_name] = {'status': 'failed', 'error': str(e)}
        
        self.cross_validation_results = cv_results
        return cv_results
    
    def analyze_prediction_patterns(self, models: Dict[str, Any],
                                   X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction patterns and errors"""
        logger.info("Analyzing prediction patterns and errors...")
        
        pattern_analysis = {}
        
        for model_name, model_info in models.items():
            if model_info['status'] != 'success' or model_info['model'] is None:
                pattern_analysis[model_name] = {'status': 'skipped', 'reason': 'Model not available'}
                continue
            
            try:
                logger.info(f"Analyzing patterns for {model_name}...")
                model = model_info['model']
                
                # Make predictions
                if len(self.target_columns) > 1:
                    y_pred = model.predict(X_val)
                else:
                    target_col = self.target_columns[0]
                    y_pred = model.predict(X_val).reshape(-1, 1)
                
                # Analyze patterns for each target
                target_patterns = {}
                for i, target in enumerate(self.target_columns):
                    if len(self.target_columns) > 1:
                        target_pred = y_pred[:, i]
                        target_true = y_val[target]
                    else:
                        target_pred = y_pred.flatten()
                        target_true = y_val[target]
                    
                    # Calculate errors
                    errors = target_true - target_pred
                    abs_errors = np.abs(errors)
                    
                    # Error statistics
                    target_patterns[target] = {
                        'mean_error': np.mean(errors),
                        'std_error': np.std(errors),
                        'mean_abs_error': np.mean(abs_errors),
                        'max_error': np.max(abs_errors),
                        'error_distribution': {
                            'q25': np.percentile(errors, 25),
                            'q50': np.percentile(errors, 50),
                            'q75': np.percentile(errors, 75)
                        },
                        'prediction_range': {
                            'min_pred': np.min(target_pred),
                            'max_pred': np.max(target_pred),
                            'min_true': np.min(target_true),
                            'max_true': np.max(target_true)
                        },
                        'correlation': np.corrcoef(target_true, target_pred)[0, 1]
                    }
                
                # Overall pattern analysis
                pattern_analysis[model_name] = {
                    'status': 'success',
                    'target_patterns': target_patterns,
                    'overall_metrics': {
                        'mean_weighted_mae': self._calculate_weighted_mae(y_val, y_pred),
                        'prediction_consistency': self._calculate_prediction_consistency(y_val, y_pred)
                    }
                }
                
            except Exception as e:
                logger.error(f"❌ Pattern analysis failed for {model_name}: {e}")
                pattern_analysis[model_name] = {'status': 'failed', 'error': str(e)}
        
        return pattern_analysis
    
    def implement_model_interpretability(self, models: Dict[str, Any],
                                       X_train: pd.DataFrame, y_train: pd.DataFrame,
                                       feature_names: List[str]) -> Dict[str, Any]:
        """Implement model interpretability analysis"""
        logger.info("Implementing model interpretability...")
        
        interpretability_results = {}
        
        for model_name, model_info in models.items():
            if model_info['status'] != 'success' or model_info['model'] is None:
                interpretability_results[model_name] = {'status': 'skipped', 'reason': 'Model not available'}
                continue
            
            try:
                logger.info(f"Analyzing interpretability for {model_name}...")
                model = model_info['model']
                
                # Feature importance analysis
                feature_importance = self._extract_feature_importance(model, model_name, X_train, y_train)
                
                # Model complexity assessment
                model_complexity = self._assess_model_complexity(model, model_name)
                
                interpretability_results[model_name] = {
                    'status': 'success',
                    'feature_importance': feature_importance,
                    'model_complexity': model_complexity
                }
                
            except Exception as e:
                logger.error(f"❌ Interpretability analysis failed for {model_name}: {e}")
                interpretability_results[model_name] = {'status': 'failed', 'error': str(e)}
        
        self.interpretability_results = interpretability_results
        return interpretability_results
    
    def _extract_feature_importance(self, model: Any, model_name: str,
                                   X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, Any]:
        """Extract feature importance from different model types"""
        
        if model_name == 'random_forest':
            if hasattr(model, 'feature_importances_'):
                return {
                    'method': 'feature_importances_',
                    'importance_scores': model.feature_importances_.tolist()
                }
        
        elif model_name == 'xgboost':
            if hasattr(model, 'feature_importances_'):
                return {
                    'method': 'feature_importances_',
                    'importance_scores': model.feature_importances_.tolist()
                }
        
        elif model_name in ['linear_regression', 'ridge_regression', 'lasso_regression']:
            if hasattr(model, 'coef_'):
                return {
                    'method': 'coefficients',
                    'importance_scores': np.abs(model.coef_).tolist()
                }
        
        # Default: use permutation importance
        try:
            from sklearn.inspection import permutation_importance
            if len(self.target_columns) > 1:
                target_col = self.target_columns[0]
                perm_importance = permutation_importance(model, X_train, y_train[target_col], 
                                                      n_repeats=5, random_state=self.random_state)
            else:
                perm_importance = permutation_importance(model, X_train, y_train, 
                                                      n_repeats=5, random_state=self.random_state)
            
            return {
                'method': 'permutation_importance',
                'importance_scores': perm_importance.importances_mean.tolist()
            }
        except Exception:
            return {
                'method': 'not_available',
                'importance_scores': [0.0] * X_train.shape[1]
            }
    
    def _assess_model_complexity(self, model: Any, model_name: str) -> Dict[str, Any]:
        """Assess model complexity and interpretability"""
        complexity = {
            'model_type': model_name,
            'is_interpretable': False,
            'complexity_score': 0
        }
        
        if model_name in ['linear_regression', 'ridge_regression', 'lasso_regression']:
            complexity['is_interpretable'] = True
            complexity['complexity_score'] = 1
            complexity['interpretability_features'] = ['linear_coefficients', 'feature_importance']
        
        elif model_name in ['random_forest', 'xgboost']:
            complexity['is_interpretable'] = True
            complexity['complexity_score'] = 2
            complexity['interpretability_features'] = ['feature_importance']
        
        elif model_name == 'knn':
            complexity['is_interpretable'] = True
            complexity['complexity_score'] = 2
            complexity['interpretability_features'] = ['neighbor_analysis']
        
        elif model_name == 'deep_learning':
            complexity['is_interpretable'] = False
            complexity['complexity_score'] = 5
            complexity['interpretability_features'] = ['feature_importance']
        
        return complexity
    
    def _calculate_weighted_mae(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
        """Calculate weighted Mean Absolute Error"""
        weights = {'Tg': 1.0, 'FFV': 1.0, 'Tc': 1.0, 'Density': 1.0, 'Rg': 1.0}
        total_mae = 0
        total_weight = 0
        
        for i, target in enumerate(self.target_columns):
            if target in y_true.columns:
                target_true = y_true[target]
                target_pred = y_pred[:, i] if y_pred.ndim > 1 else y_pred
                mae = np.mean(np.abs(target_true - target_pred))
                weight = weights.get(target, 1.0)
                total_mae += mae * weight
                total_weight += weight
        
        return total_mae / total_weight if total_weight > 0 else 0
    
    def _calculate_prediction_consistency(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
        """Calculate prediction consistency across targets"""
        correlations = []
        
        for i, target in enumerate(self.target_columns):
            if target in y_true.columns:
                target_true = y_true[target]
                target_pred = y_pred[:, i] if y_pred.ndim > 1 else y_pred
                corr = np.corrcoef(target_true, target_pred)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def create_evaluation_report(self, models: Dict[str, Any],
                                cv_results: Dict[str, Any],
                                pattern_analysis: Dict[str, Any],
                                interpretability_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive evaluation report"""
        logger.info("Creating comprehensive evaluation report...")
        
        # Overall model ranking
        model_rankings = self._rank_models_by_performance(models, cv_results)
        
        # Error analysis summary
        error_summary = self._summarize_errors(pattern_analysis)
        
        # Interpretability summary
        interpretability_summary = self._summarize_interpretability(interpretability_results)
        
        # Create comprehensive report
        evaluation_report = {
            'timestamp': datetime.now().isoformat(),
            'model_rankings': model_rankings,
            'cross_validation_summary': self._summarize_cv_results(cv_results),
            'error_analysis': error_summary,
            'interpretability_analysis': interpretability_summary,
            'recommendations': self._generate_recommendations(
                model_rankings, error_summary, interpretability_summary
            )
        }
        
        return evaluation_report
    
    def _rank_models_by_performance(self, models: Dict[str, Any], 
                                   cv_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank models by overall performance"""
        rankings = []
        
        for model_name, cv_result in cv_results.items():
            if cv_result.get('status') == 'success' and 'mean_scores' in cv_result:
                mean_scores = cv_result['mean_scores']
                weighted_mae = mean_scores.get('weighted_mae', float('inf'))
                
                rankings.append({
                    'model_name': model_name,
                    'weighted_mae': weighted_mae,
                    'status': 'success'
                })
        
        # Sort by weighted MAE (lower is better)
        rankings.sort(key=lambda x: x['weighted_mae'])
        
        # Add rank
        for i, ranking in enumerate(rankings):
            ranking['rank'] = i + 1
        
        return rankings
    
    def _summarize_cv_results(self, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize cross-validation results"""
        successful_cv = {name: result for name, result in cv_results.items() 
                        if result.get('status') == 'success'}
        
        if not successful_cv:
            return {'status': 'no_successful_cv'}
        
        # Calculate overall statistics
        all_weighted_maes = []
        for result in successful_cv.values():
            if 'mean_scores' in result and 'weighted_mae' in result['mean_scores']:
                all_weighted_maes.append(result['mean_scores']['weighted_mae'])
        
        return {
            'total_models': len(successful_cv),
            'mean_weighted_mae': np.mean(all_weighted_maes),
            'std_weighted_mae': np.std(all_weighted_maes),
            'best_weighted_mae': np.min(all_weighted_maes),
            'worst_weighted_mae': np.max(all_weighted_maes)
        }
    
    def _summarize_errors(self, pattern_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize error patterns across models"""
        successful_analysis = {name: result for name, result in pattern_analysis.items() 
                             if result.get('status') == 'success'}
        
        if not successful_analysis:
            return {'status': 'no_successful_analysis'}
        
        # Aggregate error statistics across targets
        target_error_summary = {}
        for target in self.target_columns:
            target_errors = []
            for result in successful_analysis.values():
                if target in result.get('target_patterns', {}):
                    target_errors.append(result['target_patterns'][target]['mean_abs_error'])
            
            if target_errors:
                target_error_summary[target] = {
                    'mean_mae': np.mean(target_errors),
                    'std_mae': np.std(target_errors),
                    'best_mae': np.min(target_errors),
                    'worst_mae': np.max(target_errors)
                }
        
        return {
            'total_models': len(successful_analysis),
            'target_error_summary': target_error_summary
        }
    
    def _summarize_interpretability(self, interpretability_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize model interpretability"""
        successful_interpretability = {name: result for name, result in interpretability_results.items() 
                                     if result.get('status') == 'success'}
        
        if not successful_interpretability:
            return {'status': 'no_successful_interpretability'}
        
        interpretability_summary = {
            'total_models': len(successful_interpretability),
            'interpretable_models': 0,
            'complexity_distribution': {},
            'feature_importance_available': 0
        }
        
        for result in successful_interpretability.values():
            complexity = result.get('model_complexity', {})
            if complexity.get('is_interpretable', False):
                interpretability_summary['interpretable_models'] += 1
            
            complexity_score = complexity.get('complexity_score', 0)
            interpretability_summary['complexity_distribution'][complexity_score] = \
                interpretability_summary['complexity_distribution'].get(complexity_score, 0) + 1
            
            if 'feature_importance' in result:
                interpretability_summary['feature_importance_available'] += 1
        
        return interpretability_summary
    
    def _generate_recommendations(self, model_rankings: List[Dict[str, Any]],
                                 error_summary: Dict[str, Any],
                                 interpretability_summary: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on evaluation results"""
        recommendations = []
        
        # Model selection recommendations
        if model_rankings:
            best_model = model_rankings[0]
            recommendations.append(f"Use {best_model['model_name']} as primary model (Rank 1, Weighted MAE: {best_model['weighted_mae']:.4f})")
            
            if len(model_rankings) > 1:
                second_best = model_rankings[1]
                recommendations.append(f"Consider {second_best['model_name']} as backup model (Rank 2, Weighted MAE: {second_best['weighted_mae']:.4f})")
        
        # Error analysis recommendations
        if 'target_error_summary' in error_summary:
            target_errors = error_summary['target_error_summary']
            if target_errors:
                worst_target = max(target_errors.items(), key=lambda x: x[1]['mean_mae'])
                recommendations.append(f"Focus improvement efforts on {worst_target[0]} (highest mean MAE: {worst_target[1]['mean_mae']:.4f})")
        
        # General recommendations
        recommendations.extend([
            "Perform feature engineering to improve model performance",
            "Consider ensemble methods to combine best performing models",
            "Implement cross-validation in production to monitor model drift",
            "Regularly retrain models with new data to maintain performance"
        ])
        
        return recommendations
    
    def save_evaluation_results(self, evaluation_report: Dict[str, Any],
                               output_dir: str = "models") -> str:
        """Save evaluation results to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive report
        report_path = output_path / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        logger.info(f"✅ Evaluation report saved to {report_path}")
        return str(report_path)

def evaluate_polymer_models(models: Dict[str, Any],
                           X_train: pd.DataFrame, y_train: pd.DataFrame,
                           X_val: pd.DataFrame, y_val: pd.DataFrame,
                           feature_names: List[str],
                           random_state: int = 42) -> Dict[str, Any]:
    """Convenience function for complete model evaluation pipeline"""
    evaluator = PolymerModelEvaluator(random_state=random_state)
    
    # Step 1: Cross-validation
    cv_results = evaluator.perform_cross_validation(models, X_train, y_train)
    
    # Step 2: Pattern analysis
    pattern_analysis = evaluator.analyze_prediction_patterns(models, X_val, y_val)
    
    # Step 3: Model interpretability
    interpretability_results = evaluator.implement_model_interpretability(
        models, X_train, y_train, feature_names
    )
    
    # Step 4: Create comprehensive report
    evaluation_report = evaluator.create_evaluation_report(
        models, cv_results, pattern_analysis, interpretability_results
    )
    
    # Step 5: Save results
    report_path = evaluator.save_evaluation_results(evaluation_report)
    
    return {
        'cross_validation': cv_results,
        'pattern_analysis': pattern_analysis,
        'interpretability': interpretability_results,
        'evaluation_report': evaluation_report,
        'report_path': report_path
    }
