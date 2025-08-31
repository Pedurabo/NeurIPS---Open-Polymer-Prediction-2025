"""
Baseline models for polymer property prediction.
Implements various ML algorithms as starting points for the competition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import joblib
import warnings

# ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Baseline models will be limited.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available.")

logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Baseline model for polymer property prediction.
    
    This class implements various baseline models including:
    - Linear models (Linear Regression, Ridge, Lasso)
    - Tree-based models (Random Forest, Gradient Boosting)
    - Support Vector Regression
    - K-Nearest Neighbors
    - XGBoost and LightGBM (if available)
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 target_properties: Optional[List[str]] = None,
                 random_state: int = 42):
        """
        Initialize the baseline model.
        
        Args:
            model_type: Type of model to use
            target_properties: List of target properties to predict
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.target_properties = target_properties or ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.random_state = random_state
        
        # Model and scaler
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        
        # Validation scores
        self.validation_scores = {}
        
        logger.info(f"Baseline model initialized: {model_type}")
        
    def _create_model(self, model_type: str):
        """Create a model instance based on type."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for baseline models")
            
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'ridge':
            return Ridge(alpha=1.0, random_state=self.random_state)
        elif model_type == 'lasso':
            return Lasso(alpha=0.1, random_state=self.random_state)
        elif model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif model_type == 'svr':
            return SVR(kernel='rbf', C=1.0, gamma='scale')
        elif model_type == 'knn':
            return KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def fit(self, 
            X_train: pd.DataFrame, 
            y_train: pd.DataFrame,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Fit the model to training data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of validation scores
        """
        logger.info(f"Fitting {self.model_type} model...")
        
        # Initialize models and scalers for each target
        for prop in self.target_properties:
            if prop in y_train.columns:
                # Create model
                self.models[prop] = self._create_model(self.model_type)
                
                # Create scaler
                self.scalers[prop] = StandardScaler()
                
                # Scale features
                X_scaled = self.scalers[prop].fit_transform(X_train)
                
                # Fit model
                self.models[prop].fit(X_scaled, y_train[prop])
                
                logger.info(f"Fitted model for {prop}")
        
        self.is_fitted = True
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            self.validation_scores = self.evaluate(X_val, y_val)
            logger.info(f"Validation scores: {self.validation_scores}")
        
        return self.validation_scores
        
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.
        
        Args:
            X: Features for prediction
            
        Returns:
            DataFrame with predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        predictions = {}
        
        for prop in self.target_properties:
            if prop in self.models:
                # Scale features
                X_scaled = self.scalers[prop].transform(X)
                
                # Make predictions
                pred = self.models[prop].predict(X_scaled)
                predictions[prop] = pred
                
        return pd.DataFrame(predictions, index=X.index)
        
    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance.
        
        Args:
            X: Features for evaluation
            y: True targets
            
        Returns:
            Dictionary of evaluation metrics for each target
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        scores = {}
        
        for prop in self.target_properties:
            if prop in y.columns and prop in self.models:
                # Make predictions
                y_pred = self.predict(X)[prop]
                y_true = y[prop]
                
                # Calculate metrics
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                
                scores[prop] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                }
                
        return scores
        
    def cross_validate(self, 
                      X: pd.DataFrame, 
                      y: pd.DataFrame, 
                      cv: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        cv_scores = {}
        
        for prop in self.target_properties:
            if prop in y.columns:
                # Create model
                model = self._create_model(self.model_type)
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Cross-validation scores
                mae_scores = cross_val_score(
                    model, X_scaled, y[prop], 
                    scoring='neg_mean_absolute_error', 
                    cv=cv, n_jobs=-1
                )
                
                mse_scores = cross_val_score(
                    model, X_scaled, y[prop], 
                    scoring='neg_mean_squared_error', 
                    cv=cv, n_jobs=-1
                )
                
                # Convert to positive values
                mae_scores = -mae_scores
                mse_scores = -mse_scores
                rmse_scores = np.sqrt(mse_scores)
                
                cv_scores[prop] = {
                    'MAE_mean': mae_scores.mean(),
                    'MAE_std': mae_scores.std(),
                    'RMSE_mean': rmse_scores.mean(),
                    'RMSE_std': rmse_scores.std()
                }
                
                logger.info(f"{prop} CV - MAE: {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")
                
        return cv_scores
        
    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance for tree-based models.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of feature importance DataFrames for each target
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
            
        importance_dfs = {}
        
        for prop in self.target_properties:
            if prop in self.models:
                model = self.models[prop]
                
                # Check if model has feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    
                    # Create DataFrame
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    importance_dfs[prop] = importance_df
                    
        return importance_dfs
        
    def save_model(self, filepath: str) -> None:
        """
        Save the fitted model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        model_data = {
            'model_type': self.model_type,
            'target_properties': self.target_properties,
            'random_state': self.random_state,
            'models': self.models,
            'scalers': self.scalers,
            'validation_scores': self.validation_scores,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to: {filepath}")
        
    def load_model(self, filepath: str) -> None:
        """
        Load a fitted model.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model_type = model_data['model_type']
        self.target_properties = model_data['target_properties']
        self.random_state = model_data['random_state']
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.validation_scores = model_data['validation_scores']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from: {filepath}")
        
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.model_type,
            'target_properties': self.target_properties,
            'is_fitted': self.is_fitted,
            'validation_scores': self.validation_scores,
            'random_state': self.random_state
        }


def create_baseline_ensemble(X_train: pd.DataFrame,
                            y_train: pd.DataFrame,
                            X_val: pd.DataFrame,
                            y_val: pd.DataFrame,
                            target_properties: List[str]) -> Dict[str, BaselineModel]:
    """
    Create an ensemble of baseline models.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        target_properties: List of target properties
        
    Returns:
        Dictionary of fitted models
    """
    logger.info("Creating baseline ensemble...")
    
    # Model types to try
    model_types = ['random_forest', 'gradient_boosting', 'ridge', 'lasso']
    
    if XGBOOST_AVAILABLE:
        model_types.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        model_types.append('lightgbm')
    
    models = {}
    
    for model_type in model_types:
        try:
            logger.info(f"Training {model_type}...")
            
            model = BaselineModel(
                model_type=model_type,
                target_properties=target_properties
            )
            
            # Fit model
            validation_scores = model.fit(X_train, y_train, X_val, y_val)
            
            # Store model
            models[model_type] = model
            
            logger.info(f"{model_type} trained successfully")
            
        except Exception as e:
            logger.warning(f"Failed to train {model_type}: {e}")
            continue
    
    return models


def evaluate_ensemble(ensemble: Dict[str, BaselineModel],
                     X_val: pd.DataFrame,
                     y_val: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate ensemble performance.
    
    Args:
        ensemble: Dictionary of models
        X_val: Validation features
        y_val: Validation targets
        
    Returns:
        DataFrame with evaluation results
    """
    results = []
    
    for model_name, model in ensemble.items():
        try:
            scores = model.evaluate(X_val, y_val)
            
            for target, metrics in scores.items():
                results.append({
                    'model': model_name,
                    'target': target,
                    **metrics
                })
                
        except Exception as e:
            logger.warning(f"Failed to evaluate {model_name}: {e}")
            continue
    
    return pd.DataFrame(results)
