#!/usr/bin/env python3
"""
Main training script for NeurIPS Open Polymer Prediction 2025.

This script demonstrates the complete pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training
4. Prediction and submission generation

Run this script to train baseline models and generate predictions.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Add src to path
sys.path.append('src')

# Import custom modules
from data.loader import PolymerDataLoader
from data.preprocessor import PolymerDataPreprocessor
from features.engineer import PolymerFeatureEngineer
from models.baseline_model import BaselineModel, create_baseline_ensemble

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


def main():
    """Main training pipeline."""
    print("🚀 NeurIPS Open Polymer Prediction 2025 - Training Pipeline")
    print("=" * 70)
    
    # Configuration
    data_dir = "data"
    models_dir = "models"
    submissions_dir = "submissions"
    
    # Create directories if they don't exist
    Path(models_dir).mkdir(exist_ok=True)
    Path(submissions_dir).mkdir(exist_ok=True)
    
    # Step 1: Data Loading
    print("\n📊 Step 1: Loading Data")
    print("-" * 30)
    
    try:
        data_loader = PolymerDataLoader(data_dir=data_dir)
        data_files = data_loader.load_all_data()
        
        train_data = data_files.get('train')
        test_data = data_files.get('test')
        
        if train_data is None:
            print("❌ Training data not found. Please ensure train.csv is in the data directory.")
            return
            
        print(f"✓ Training data loaded: {train_data.shape}")
        if test_data is not None:
            print(f"✓ Test data loaded: {test_data.shape}")
            
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Data Preprocessing
    print("\n🔬 Step 2: Data Preprocessing")
    print("-" * 30)
    
    try:
        # Initialize preprocessor
        preprocessor = PolymerDataPreprocessor(
            use_rdkit=True,
            use_mordred=True,
            use_fingerprints=True,
            fingerprint_size=1024,  # Reduced for faster processing
            use_3d_descriptors=False
        )
        
        # Get SMILES and target data
        smiles_data = train_data['SMILES']
        target_data = train_data[['Tg', 'FFV', 'Tc', 'Density', 'Rg']]
        
        print(f"Processing {len(smiles_data)} SMILES...")
        
        # Extract features
        features_df, targets_df = preprocessor.preprocess_data(
            smiles_list=smiles_data.tolist(),
            target_data=target_data
        )
        
        print(f"✓ Features extracted: {features_df.shape}")
        print(f"✓ Targets aligned: {targets_df.shape}")
        
        # Normalize features
        features_normalized = preprocessor.normalize_features(features_df, fit=True)
        print(f"✓ Features normalized: {features_normalized.shape}")
        
        # Save preprocessor
        preprocessor.save_preprocessor(f"{models_dir}/preprocessor.joblib")
        print("✓ Preprocessor saved")
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return
    
    # Step 3: Feature Engineering
    print("\n⚙️ Step 3: Feature Engineering")
    print("-" * 30)
    
    try:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(
            create_interactions=True,
            create_polynomials=False,  # Disabled for baseline
            use_pca=False,  # Disabled for baseline
            create_statistical_features=True,
            create_domain_features=True
        )
        
        # Engineer features
        features_engineered = feature_engineer.engineer_features(features_normalized, fit=True)
        print(f"✓ Features engineered: {features_engineered.shape}")
        
        # Save feature engineer
        feature_engineer.save_engineer(f"{models_dir}/feature_engineer.joblib")
        print("✓ Feature engineer saved")
        
    except Exception as e:
        print(f"❌ Error in feature engineering: {e}")
        # Continue with original features
        features_engineered = features_normalized
        print("⚠ Continuing with original features")
    
    # Step 4: Data Splitting
    print("\n✂️ Step 4: Data Splitting")
    print("-" * 30)
    
    try:
        # Split data
        train_data_split, val_data_split = data_loader.split_train_val(val_size=0.2, random_state=42)
        
        # Get indices for splitting features and targets
        train_indices = train_data_split.index
        val_indices = val_data_split.index
        
        # Split features and targets
        X_train = features_engineered.loc[train_indices]
        X_val = features_engineered.loc[val_indices]
        y_train = targets_df.loc[train_indices]
        y_val = targets_df.loc[val_indices]
        
        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Validation set: {X_val.shape}")
        
    except Exception as e:
        print(f"❌ Error in data splitting: {e}")
        return
    
    # Step 5: Model Training
    print("\n🤖 Step 5: Model Training")
    print("-" * 30)
    
    try:
        # Get target properties
        target_properties = data_loader.get_target_properties()
        
        # Create baseline ensemble
        print("Training baseline ensemble...")
        ensemble = create_baseline_ensemble(
            X_train, y_train, X_val, y_val, target_properties
        )
        
        if not ensemble:
            print("❌ No models were trained successfully")
            return
            
        print(f"✓ Ensemble created with {len(ensemble)} models")
        
        # Evaluate ensemble
        print("\nEvaluating ensemble performance...")
        ensemble_results = evaluate_ensemble(ensemble, X_val, y_val)
        
        # Display results
        print("\n📊 Ensemble Performance Summary:")
        print(ensemble_results.groupby('model')['MAE'].mean().sort_values())
        
        # Save best model
        best_model_name = ensemble_results.groupby('model')['MAE'].mean().idxmin()
        best_model = ensemble[best_model_name]
        
        best_model.save_model(f"{models_dir}/best_baseline_model.joblib")
        print(f"✓ Best model saved: {best_model_name}")
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return
    
    # Step 6: Generate Predictions
    print("\n🔮 Step 6: Generating Predictions")
    print("-" * 30)
    
    try:
        if test_data is not None:
            # Preprocess test data
            test_smiles = data_loader.get_smiles_data(test_data)
            
            # Extract features for test data
            test_features, _ = preprocessor.preprocess_data(
                smiles_list=test_smiles.tolist(),
                target_data=None
            )
            
            # Normalize test features
            test_features_normalized = preprocessor.normalize_features(test_features, fit=False)
            
            # Engineer features for test data
            test_features_engineered = feature_engineer.engineer_features(
                test_features_normalized, fit=False
            )
            
            print(f"✓ Test features prepared: {test_features_engineered.shape}")
            
            # Make predictions
            predictions = best_model.predict(test_features_engineered)
            print(f"✓ Predictions generated: {predictions.shape}")
            
            # Create submission
            submission = test_data[['id']].copy()
            for prop in target_properties:
                if prop in predictions.columns:
                    submission[prop] = predictions[prop]
                else:
                    submission[prop] = 0.0  # Default value
            
            # Save submission
            submission_path = f"{submissions_dir}/submission_baseline.csv"
            submission.to_csv(submission_path, index=False)
            print(f"✓ Submission saved: {submission_path}")
            
            # Show submission preview
            print("\n📝 Submission Preview:")
            print(submission.head())
            
        else:
            print("⚠ No test data available for predictions")
            
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
    
    # Step 7: Summary
    print("\n🎯 Training Pipeline Complete!")
    print("=" * 70)
    print(f"📁 Models saved to: {models_dir}/")
    print(f"📁 Submissions saved to: {submissions_dir}/")
    print(f"🔬 Preprocessor saved to: {models_dir}/preprocessor.joblib")
    print(f"⚙️ Feature engineer saved to: {models_dir}/feature_engineer.joblib")
    
    print("\n📊 Next Steps:")
    print("1. Analyze model performance and feature importance")
    print("2. Experiment with different feature engineering approaches")
    print("3. Try advanced models (Neural Networks, Transformers)")
    print("4. Implement hyperparameter tuning")
    print("5. Create ensemble predictions")
    print("6. Submit to competition!")
    
    return ensemble, ensemble_results


def evaluate_ensemble(ensemble, X_val, y_val):
    """Evaluate ensemble performance."""
    from models.baseline_model import evaluate_ensemble as eval_ensemble
    return eval_ensemble(ensemble, X_val, y_val)


if __name__ == "__main__":
    try:
        ensemble, results = main()
        print("\n✅ Script completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠ Script interrupted by user")
    except Exception as e:
        print(f"\n❌ Script failed with error: {e}")
        import traceback
        traceback.print_exc()
