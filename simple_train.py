#!/usr/bin/env python3
"""
Simplified Training Script for NeurIPS Open Polymer Prediction 2025
Uses existing processed data and simplified approach
"""

import sys
import os
import logging
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Add src to path
sys.path.append('src')

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline using existing processed data."""
    print("🚀 NeurIPS Open Polymer Prediction 2025 - Simple Training Pipeline")
    print("=" * 70)
    
    # Configuration
    data_dir = "data/processed"
    models_dir = "models"
    submissions_dir = "submissions"
    
    # Create directories if they don't exist
    Path(models_dir).mkdir(exist_ok=True)
    Path(submissions_dir).mkdir(exist_ok=True)
    
    # Step 1: Load Processed Data
    print("\n📊 Step 1: Loading Processed Data")
    print("-" * 30)
    
    try:
        # Load the processed feature matrix
        feature_matrix_path = Path(data_dir) / "feature_matrix_final.csv"
        if not feature_matrix_path.exists():
            print("❌ Processed feature matrix not found. Please run the preprocessing pipeline first.")
            return None
            
        # Load features and targets
        feature_matrix = pd.read_csv(feature_matrix_path)
        print(f"✓ Feature matrix loaded: {feature_matrix.shape}")
        
        # Separate features and targets
        target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        feature_columns = [col for col in feature_matrix.columns if col not in target_columns + ['id']]
        
        X = feature_matrix[feature_columns]
        y = feature_matrix[target_columns]
        
        print(f"✓ Features: {X.shape}")
        print(f"✓ Targets: {y.shape}")
        
        # Remove any rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull().any(axis=1))
        X = X[mask]
        y = y[mask]
        
        print(f"✓ After removing missing values: {X.shape}")
        
    except Exception as e:
        print(f"❌ Error loading processed data: {e}")
        return None
    
    # Step 2: Data Splitting
    print("\n✂️ Step 2: Data Splitting")
    print("-" * 30)
    
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Validation set: {X_val.shape}")
        
    except Exception as e:
        print(f"❌ Error in data splitting: {e}")
        return None
    
    # Step 3: Model Training
    print("\n🤖 Step 3: Model Training")
    print("-" * 30)
    
    try:
        # Train individual models for each target
        models = {}
        train_scores = {}
        val_scores = {}
        
        for target in target_columns:
            print(f"\nTraining model for {target}...")
            
            # Initialize model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            model.fit(X_train, y_train[target])
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_mae = mean_absolute_error(y_train[target], train_pred)
            val_mae = mean_absolute_error(y_val[target], val_pred)
            train_r2 = r2_score(y_train[target], train_pred)
            val_r2 = r2_score(y_val[target], val_pred)
            
            models[target] = model
            train_scores[target] = {'mae': train_mae, 'r2': train_r2}
            val_scores[target] = {'mae': val_mae, 'r2': val_r2}
            
            print(f"  {target}: Train MAE={train_mae:.4f}, R²={train_r2:.4f}")
            print(f"  {target}: Val MAE={val_mae:.4f}, R²={val_r2:.4f}")
        
        # Calculate overall scores
        overall_train_mae = np.mean([scores['mae'] for scores in train_scores.values()])
        overall_val_mae = np.mean([scores['mae'] for scores in val_scores.values()])
        overall_train_r2 = np.mean([scores['r2'] for scores in train_scores.values()])
        overall_val_r2 = np.mean([scores['r2'] for scores in val_scores.values()])
        
        print(f"\n📊 Overall Performance:")
        print(f"  Training: MAE={overall_train_mae:.4f}, R²={overall_train_r2:.4f}")
        print(f"  Validation: MAE={overall_val_mae:.4f}, R²={overall_val_r2:.4f}")
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return None
    
    # Step 4: Save Models
    print("\n💾 Step 4: Saving Models")
    print("-" * 30)
    
    try:
        # Save individual models
        for target, model in models.items():
            model_path = Path(models_dir) / f"rf_model_{target}.joblib"
            joblib.dump(model, model_path)
            print(f"✓ Saved {target} model: {model_path}")
        
        # Save ensemble model (all models together)
        ensemble_path = Path(models_dir) / "rf_ensemble.joblib"
        joblib.dump(models, ensemble_path)
        print(f"✓ Saved ensemble model: {ensemble_path}")
        
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return None
    
    # Step 5: Generate Predictions
    print("\n🔮 Step 5: Generating Predictions")
    print("-" * 30)
    
    try:
        # Load test data if available
        test_matrix_path = Path(data_dir) / "feature_matrix_validation.csv"
        if test_matrix_path.exists():
            test_matrix = pd.read_csv(test_matrix_path)
            test_features = test_matrix[feature_columns]
            
            # Generate predictions
            predictions = {}
            for target in target_columns:
                if target in models:
                    pred = models[target].predict(test_features)
                    predictions[target] = pred
            
            # Create submission DataFrame
            submission_df = pd.DataFrame(predictions)
            submission_df.insert(0, 'id', test_matrix['id'] if 'id' in test_matrix.columns else range(len(submission_df)))
            
            # Save submission
            submission_path = Path(submissions_dir) / "submission_rf.csv"
            submission_df.to_csv(submission_path, index=False)
            print(f"✓ Submission saved: {submission_path}")
            print(f"✓ Submission shape: {submission_df.shape}")
            
        else:
            print("⚠ Test data not found, skipping prediction generation")
            
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return None
    
    print("\n🎉 Training pipeline completed successfully!")
    return models

if __name__ == "__main__":
    try:
        models = main()
        if models is None:
            print("\n❌ Script failed with error")
            sys.exit(1)
        else:
            print("\n✅ Script completed successfully!")
            sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
