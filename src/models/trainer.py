"""
Model Training for NeurIPS Open Polymer Prediction 2025
Cluster 4: Train baseline ML models, implement deep learning, and optimize performance
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

class PolymerModelTrainer:
    """Comprehensive model trainer for polymer property prediction"""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model trainer"""
        self.random_state = random_state
        np.random.seed(random_state)
        self.target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.training_results = {}
        self.model_performance = {}
        self.model_configs = self._get_default_model_configs()
        
    def _get_default_model_configs(self) -> Dict[str, Dict]:
        """Get default model configurations"""
        return {
            'linear_regression': {'model_type': 'linear', 'params': {}},
            'ridge_regression': {'model_type': 'linear', 'params': {'alpha': 1.0}},
            'lasso_regression': {'model_type': 'linear', 'params': {'alpha': 0.1}},
            'random_forest': {'model_type': 'ensemble', 'params': {'n_estimators': 100, 'max_depth': 10, 'random_state': self.random_state}},
            'xgboost': {'model_type': 'gradient_boosting', 'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': self.random_state}},
            'lightgbm': {'model_type': 'gradient_boosting', 'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'random_state': self.random_state}},
            'svr': {'model_type': 'svm', 'params': {'kernel': 'rbf', 'C': 1.0}},
            'knn': {'model_type': 'neighbors', 'params': {'n_neighbors': 5}}
        }
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                             X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, Any]:
        """Train baseline machine learning models"""
        logger.info("Training baseline machine learning models...")
        results = {}
        
        for model_name, config in self.model_configs.items():
            try:
                logger.info(f"Training {model_name}...")
                model, train_time, val_metrics = self._train_single_model(
                    model_name, config, X_train, y_train, X_val, y_val
                )
                
                results[model_name] = {
                    'model': model, 'config': config, 'train_time': train_time,
                    'validation_metrics': val_metrics, 'status': 'success'
                }
                logger.info(f"✅ {model_name} trained successfully in {train_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                results[model_name] = {
                    'model': None, 'config': config, 'train_time': 0,
                    'validation_metrics': {}, 'status': 'failed', 'error': str(e)
                }
        
        self.training_results = results
        return results
    
    def _train_single_model(self, model_name: str, config: Dict,
                           X_train: pd.DataFrame, y_train: pd.DataFrame,
                           X_val: pd.DataFrame, y_val: pd.DataFrame) -> Tuple[Any, float, Dict]:
        """Train a single model"""
        import time
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        start_time = time.time()
        model = self._create_model(model_name, config)
        
        # Handle multi-target regression
        if len(self.target_columns) > 1:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        else:
            target_col = self.target_columns[0]
            model.fit(X_train, y_train[target_col])
            y_pred = model.predict(X_val).reshape(-1, 1)
        
        train_time = time.time() - start_time
        
        # Calculate metrics for each target
        val_metrics = {}
        for i, target in enumerate(self.target_columns):
            if len(self.target_columns) > 1:
                target_pred = y_pred[:, i]
                target_true = y_val[target]
            else:
                target_pred = y_pred.flatten()
                target_true = y_val[target]
            
            mae = mean_absolute_error(target_true, target_pred)
            mse = mean_squared_error(target_true, target_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(target_true, target_pred)
            
            val_metrics[target] = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
        
        val_metrics['weighted_mae'] = self._calculate_weighted_mae(y_val, y_pred)
        return model, train_time, val_metrics
    
    def _create_model(self, model_name: str, config: Dict) -> Any:
        """Create a model instance based on name and config"""
        if model_name == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**config['params'])
        elif model_name == 'ridge_regression':
            from sklearn.linear_model import Ridge
            return Ridge(**config['params'])
        elif model_name == 'lasso_regression':
            from sklearn.linear_model import Lasso
            return Lasso(**config['params'])
        elif model_name == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**config['params'])
        elif model_name == 'xgboost':
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(**config['params'])
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    n_estimators=config['params']['n_estimators'],
                    max_depth=config['params']['max_depth'],
                    learning_rate=config['params']['learning_rate'],
                    random_state=config['params']['random_state']
                )
        elif model_name == 'lightgbm':
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(**config['params'])
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(
                    n_estimators=config['params']['n_estimators'],
                    max_depth=config['params']['max_depth'],
                    learning_rate=config['params']['learning_rate'],
                    random_state=config['params']['random_state']
                )
        elif model_name == 'svr':
            from sklearn.svm import SVR
            return SVR(**config['params'])
        elif model_name == 'knn':
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(**config['params'])
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def _calculate_weighted_mae(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
        """Calculate weighted Mean Absolute Error (competition metric)"""
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
    
    def implement_deep_learning(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                               X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, Any]:
        """Implement deep learning approaches"""
        logger.info("Implementing deep learning with MLP...")
        
        try:
            return self._train_mlp(X_train, y_train, X_val, y_val)
        except Exception as e:
            logger.error(f"❌ Deep learning implementation failed: {e}")
            return {'model': None, 'model_type': 'mlp', 'status': 'failed', 'error': str(e)}
    
    def _train_mlp(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, Any]:
        """Train Multi-Layer Perceptron"""
        try:
            from sklearn.neural_network import MLPRegressor
            
            model = MLPRegressor(
                hidden_layer_sizes=(512, 256, 128),
                activation='relu', solver='adam', alpha=0.001,
                batch_size=32, learning_rate='adaptive',
                max_iter=200, random_state=self.random_state
            )
            
            # Train model
            if len(self.target_columns) > 1:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            else:
                target_col = self.target_columns[0]
                model.fit(X_train, y_train[target_col])
                y_pred = model.predict(X_val).reshape(-1, 1)
            
            # Calculate metrics
            val_metrics = self._calculate_deep_learning_metrics(y_val, y_pred)
            
            return {
                'model': model, 'model_type': 'mlp_sklearn',
                'architecture': 'MLP(512, 256, 128)',
                'validation_metrics': val_metrics, 'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"❌ Sklearn MLP training failed: {e}")
            return {'model': None, 'model_type': 'mlp_sklearn', 'status': 'failed', 'error': str(e)}
    
    def _calculate_deep_learning_metrics(self, y_true: pd.DataFrame, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate metrics for deep learning models"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        metrics = {}
        for i, target in enumerate(self.target_columns):
            if target in y_true.columns:
                target_true = y_true[target]
                target_pred = y_pred[:, i] if y_pred.ndim > 1 else y_pred
                
                mae = mean_absolute_error(target_true, target_pred)
                mse = mean_squared_error(target_true, target_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(target_true, target_pred)
                
                metrics[target] = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
        
        metrics['weighted_mae'] = self._calculate_weighted_mae(y_true, y_pred)
        return metrics
    
    def evaluate_model_performance(self, models: Dict[str, Any],
                                 X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate performance of all trained models"""
        logger.info("Evaluating model performance...")
        performance = {}
        
        for model_name, model_info in models.items():
            if model_info['status'] == 'success' and model_info['model'] is not None:
                try:
                    model = model_info['model']
                    y_pred = model.predict(X_val)
                    metrics = self._calculate_deep_learning_metrics(y_val, y_pred)
                    
                    performance[model_name] = {
                        'metrics': metrics,
                        'config': model_info.get('config', {}),
                        'train_time': model_info.get('train_time', 0),
                        'model_type': model_info.get('model_type', 'unknown')
                    }
                except Exception as e:
                    logger.error(f"❌ Error evaluating {model_name}: {e}")
                    performance[model_name] = {'metrics': {}, 'error': str(e), 'status': 'evaluation_failed'}
            else:
                performance[model_name] = {
                    'metrics': {}, 'status': 'not_trained',
                    'error': model_info.get('error', 'Training failed')
                }
        
        self.model_performance = performance
        return performance
    
    def create_ensemble_model(self, models: Dict[str, Any],
                             X_train: pd.DataFrame, y_train: pd.DataFrame,
                             X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, Any]:
        """Create ensemble model from trained models"""
        logger.info("Creating ensemble model using voting...")
        
        successful_models = {
            name: info for name, info in models.items()
            if info['status'] == 'success' and info['model'] is not None
        }
        
        if len(successful_models) < 2:
            return {'status': 'failed', 'reason': 'Insufficient successful models'}
        
        try:
            from sklearn.ensemble import VotingRegressor
            
            # Prepare estimators
            estimators = [(name, info['model']) for name, info in successful_models.items()]
            
            # Create voting ensemble
            ensemble = VotingRegressor(estimators=estimators)
            
            # Make predictions
            y_pred = ensemble.predict(X_val)
            
            # Calculate metrics
            metrics = self._calculate_deep_learning_metrics(y_val, y_pred)
            
            return {
                'status': 'success', 'ensemble_type': 'voting',
                'base_models': list(successful_models.keys()),
                'ensemble_model': ensemble, 'validation_metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Ensemble creation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def save_models(self, output_dir: str = "models") -> Dict[str, str]:
        """Save all trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save baseline models
        for name, result in self.training_results.items():
            if result['status'] == 'success' and result['model'] is not None:
                try:
                    model_path = output_path / f"{name}_model.pkl"
                    with open(model_path, 'wb') as f:
                        pickle.dump(result['model'], f)
                    saved_paths[name] = str(model_path)
                    logger.info(f"✅ Saved {name} model to {model_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to save {name} model: {e}")
        
        # Save training results
        results_path = output_path / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        saved_paths['training_results'] = str(results_path)
        
        return saved_paths
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'training_results': self.training_results,
            'model_performance': self.model_performance,
            'target_columns': self.target_columns,
            'random_state': self.random_state,
            'timestamp': datetime.now().isoformat()
        }

def train_polymer_models(X_train: pd.DataFrame, y_train: pd.DataFrame,
                        X_val: pd.DataFrame, y_val: pd.DataFrame,
                        random_state: int = 42) -> Dict[str, Any]:
    """Convenience function for complete model training pipeline"""
    trainer = PolymerModelTrainer(random_state=random_state)
    
    # Train baseline models
    baseline_results = trainer.train_baseline_models(X_train, y_train, X_val, y_val)
    
    # Implement deep learning
    deep_learning_results = trainer.implement_deep_learning(X_train, y_train, X_val, y_val)
    
    # Evaluate all models
    all_models = {**baseline_results, 'deep_learning': deep_learning_results}
    performance = trainer.evaluate_model_performance(all_models, X_val, y_val)
    
    # Create ensemble
    ensemble_results = trainer.create_ensemble_model(all_models, X_train, y_train, X_val, y_val)
    
    # Get complete summary
    training_summary = trainer.get_training_summary()
    
    return {
        'baseline_models': baseline_results,
        'deep_learning': deep_learning_results,
        'performance': performance,
        'ensemble': ensemble_results,
        'summary': training_summary
    }
