#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - Feature Engineering
Cluster 3: Extract molecular descriptors, generate fingerprints, and create features

This script implements the third phase of the CRISP-DM methodology:
1. Extract molecular descriptors from SMILES strings
2. Generate Morgan fingerprints (or simplified alternatives)
3. Create custom polymer-specific features
4. Implement feature selection
5. Prepare final feature matrix for machine learning

Based on Cluster 2 results:
- Clean, preprocessed data with 0% missing values
- Standardized features ready for engineering
- Train/validation split established
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

# Standard data science imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Import our custom modules
from data.loader import PolymerDataLoader
from features.engineer import PolymerFeatureEngineer, engineer_polymer_features

def main():
    """Main feature engineering function"""
    print("=" * 80)
    print("NEURIPS OPEN POLYMER PREDICTION 2025 - FEATURE ENGINEERING")
    print("CLUSTER 3: Extract Molecular Descriptors & Generate Features")
    print("=" * 80)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    print("\n🔍 Initializing feature engineering tools...")
    engineer = PolymerFeatureEngineer(random_state=42)
    
    # Load preprocessed data
    print("\n📊 Loading preprocessed data...")
    processed_dir = "data/processed"
    
    if not os.path.exists(processed_dir):
        print(f"❌ Processed data directory not found: {processed_dir}")
        print("Please run Cluster 2 (Data Preparation) first.")
        return
    
    # Check for preprocessed files
    train_file = os.path.join(processed_dir, "train_preprocessed.csv")
    val_file = os.path.join(processed_dir, "val_preprocessed.csv")
    
    if not os.path.exists(train_file):
        print(f"❌ Preprocessed training data not found: {train_file}")
        print("Please run Cluster 2 (Data Preparation) first.")
        return
    
    # Load preprocessed data
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file) if os.path.exists(val_file) else None
    
    print(f"✅ Training data loaded: {train_data.shape}")
    if val_data is not None:
        print(f"✅ Validation data loaded: {val_data.shape}")
    
    # Task 3.1: Extract Molecular Descriptors
    print("\n" + "=" * 60)
    print("TASK 3.1: EXTRACTING MOLECULAR DESCRIPTORS")
    print("=" * 60)
    
    print("\n🧪 Extracting molecular descriptors from SMILES...")
    molecular_descriptors = engineer.extract_molecular_descriptors(train_data['SMILES'])
    
    print(f"\n✅ Molecular descriptors extracted: {molecular_descriptors.shape}")
    print(f"  Descriptors created: {list(molecular_descriptors.columns)}")
    
    # Show sample descriptors
    print(f"\n📊 Sample molecular descriptors:")
    print(molecular_descriptors.head(3).round(4))
    
    # Task 3.2: Generate Morgan Fingerprints
    print("\n" + "=" * 60)
    print("TASK 3.2: GENERATING MORGAN FINGERPRINTS")
    print("=" * 60)
    
    print("\n🔬 Generating Morgan fingerprints...")
    morgan_fingerprints = engineer.generate_morgan_fingerprints(train_data['SMILES'])
    
    print(f"\n✅ Morgan fingerprints generated: {morgan_fingerprints.shape}")
    print(f"  Fingerprint bits: {morgan_fingerprints.shape[1]}")
    
    # Show fingerprint statistics
    print(f"\n📊 Fingerprint statistics:")
    print(f"  Non-zero bits per sample: {morgan_fingerprints.sum(axis=1).describe().round(2)}")
    print(f"  Bit density: {(morgan_fingerprints.sum().sum() / (morgan_fingerprints.shape[0] * morgan_fingerprints.shape[1]) * 100):.2f}%")
    
    # Task 3.3: Create Custom Polymer Features
    print("\n" + "=" * 60)
    print("TASK 3.3: CREATING CUSTOM POLYMER FEATURES")
    print("=" * 60)
    
    print("\n🏗️ Creating custom polymer-specific features...")
    custom_features = engineer.create_custom_polymer_features(train_data['SMILES'], molecular_descriptors)
    
    print(f"\n✅ Custom polymer features created: {custom_features.shape}")
    print(f"  Features created: {list(custom_features.columns)}")
    
    # Show sample custom features
    print(f"\n📊 Sample custom polymer features:")
    print(custom_features.head(3).round(4))
    
    # Task 3.4: Create Complete Feature Matrix
    print("\n" + "=" * 60)
    print("TASK 3.4: CREATING COMPLETE FEATURE MATRIX")
    print("=" * 60)
    
    print("\n🔧 Creating complete feature matrix...")
    feature_matrix, feature_info = engineer.create_final_feature_matrix(
        train_data, smiles_column='SMILES', include_targets=True
    )
    
    print(f"\n✅ Complete feature matrix created: {feature_matrix.shape}")
    print(f"  Total features: {feature_info['total_features']}")
    print(f"  Molecular descriptors: {feature_info['molecular_descriptors']}")
    print(f"  Morgan fingerprints: {feature_info['morgan_fingerprints']}")
    print(f"  Custom features: {feature_info['custom_features']}")
    print(f"  Targets included: {feature_info['targets_included']}")
    
    # Task 3.5: Implement Feature Selection
    print("\n" + "=" * 60)
    print("TASK 3.5: IMPLEMENTING FEATURE SELECTION")
    print("=" * 60)
    
    print("\n🎯 Implementing feature selection...")
    
    # Prepare target data for feature selection
    target_data = train_data[engineer.target_columns]
    
    # Implement feature selection
    selected_features, selected_names = engineer.implement_feature_selection(
        feature_matrix.drop(columns=engineer.target_columns),  # Exclude targets
        target_data,
        method='correlation',
        threshold=0.01,
        max_features=1000
    )
    
    print(f"\n✅ Feature selection completed!")
    print(f"  Original features: {len(feature_matrix.columns) - len(engineer.target_columns)}")
    print(f"  Selected features: {len(selected_names)}")
    print(f"  Reduction: {((len(feature_matrix.columns) - len(engineer.target_columns) - len(selected_names)) / (len(feature_matrix.columns) - len(engineer.target_columns)) * 100):.1f}%")
    
    # Create final selected feature matrix
    final_feature_matrix = pd.concat([
        selected_features,
        train_data[engineer.target_columns]
    ], axis=1)
    
    print(f"\n📊 Final feature matrix: {final_feature_matrix.shape}")
    
    # Task 3.6: Feature Analysis and Visualization
    print("\n" + "=" * 60)
    print("TASK 3.6: FEATURE ANALYSIS & VISUALIZATION")
    print("=" * 60)
    
    print("\n📈 Analyzing feature characteristics...")
    
    # Feature type breakdown
    feature_types = {
        'Molecular Descriptors': [col for col in selected_names if col in molecular_descriptors.columns],
        'Morgan Fingerprints': [col for col in selected_names if col in morgan_fingerprints.columns],
        'Custom Polymer Features': [col for col in selected_names if col in custom_features.columns]
    }
    
    print(f"\n📊 Selected feature breakdown:")
    for feature_type, features in feature_types.items():
        print(f"  {feature_type}: {len(features)} features")
    
    # Feature importance analysis (correlation with targets)
    print(f"\n🎯 Feature importance analysis:")
    for target in engineer.target_columns:
        if target in final_feature_matrix.columns:
            correlations = final_feature_matrix[selected_names].corrwith(final_feature_matrix[target]).abs()
            top_features = correlations.nlargest(5)
            print(f"\n  Top features for {target}:")
            for feature, corr in top_features.items():
                print(f"    {feature}: {corr:.4f}")
    
    # Task 3.7: Save Feature Matrix and Summary
    print("\n" + "=" * 60)
    print("TASK 3.7: SAVING FEATURE MATRIX & SUMMARY")
    print("=" * 60)
    
    print("\n💾 Saving feature engineering results...")
    
    # Save final feature matrix
    feature_matrix_path = engineer.save_feature_matrix(
        final_feature_matrix, 
        output_dir="data/processed",
        filename="feature_matrix_final.csv"
    )
    print(f"  ✅ Feature matrix saved: {feature_matrix_path}")
    
    # Save training and validation feature matrices
    if val_data is not None:
        # Create validation feature matrix (without targets for inference)
        val_feature_matrix, _ = engineer.create_final_feature_matrix(
            val_data, smiles_column='SMILES', include_targets=False
        )
        
        # Select same features for validation
        val_selected = val_feature_matrix[selected_names]
        val_final = pd.concat([
            val_selected,
            val_data[engineer.target_columns]
        ], axis=1)
        
        val_path = engineer.save_feature_matrix(
            val_final,
            output_dir="data/processed",
            filename="feature_matrix_validation.csv"
        )
        print(f"  ✅ Validation feature matrix saved: {val_path}")
    
    # Save feature engineering summary
    feature_summary = engineer.get_feature_engineering_summary()
    feature_summary['feature_types_breakdown'] = feature_types
    feature_summary['selected_feature_names'] = selected_names
    
    summary_path = os.path.join(processed_dir, "feature_engineering_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(feature_summary, f, indent=2, default=str)
    print(f"  ✅ Feature engineering summary saved: {summary_path}")
    
    # Task 3.8: Generate Feature Engineering Report
    print("\n" + "=" * 60)
    print("TASK 3.8: FEATURE ENGINEERING REPORT")
    print("=" * 60)
    
    print("\n📋 GENERATING FEATURE ENGINEERING REPORT")
    print("=" * 60)
    
    print(f"\n🏗️ FEATURE ENGINEERING SUMMARY:")
    print(f"  Input data shape: {train_data.shape}")
    print(f"  Final feature matrix: {final_feature_matrix.shape}")
    print(f"  Features created: {feature_info['total_features']}")
    print(f"  Features selected: {len(selected_names)}")
    
    print(f"\n🧪 FEATURE TYPES CREATED:")
    print(f"  Molecular descriptors: {feature_info['molecular_descriptors']}")
    print(f"  Morgan fingerprints: {feature_info['morgan_fingerprints']}")
    print(f"  Custom polymer features: {feature_info['custom_features']}")
    
    print(f"\n🎯 FEATURE SELECTION RESULTS:")
    print(f"  Selection method: Correlation-based")
    print(f"  Correlation threshold: 0.01")
    print(f"  Maximum features: 1000")
    print(f"  Features selected: {len(selected_names)}")
    
    print(f"\n📊 FEATURE BREAKDOWN BY TYPE:")
    for feature_type, features in feature_types.items():
        print(f"  {feature_type}: {len(features)} features")
    
    print(f"\n💾 OUTPUT FILES:")
    print(f"  Training feature matrix: {feature_matrix_path}")
    if val_data is not None:
        print(f"  Validation feature matrix: {val_path}")
    print(f"  Feature engineering summary: {summary_path}")
    
    # Task 3.9: Next Steps for Cluster 4
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS FOR CLUSTER 4 (MODEL TRAINING)")
    print("=" * 60)
    
    print("\n📚 Ready for Model Training:")
    print("1. ✅ Molecular descriptors extracted")
    print("2. ✅ Morgan fingerprints generated")
    print("3. ✅ Custom polymer features created")
    print("4. ✅ Feature selection implemented")
    print("5. ✅ Final feature matrix prepared")
    
    print("\n🎯 Next Phase Tasks:")
    print("1. Train baseline machine learning models")
    print("2. Implement deep learning approaches")
    print("3. Perform hyperparameter optimization")
    print("4. Evaluate model performance")
    print("5. Create ensemble models")
    
    print("\n💾 Data Files Ready:")
    print(f"  Training features: {feature_matrix_path}")
    if val_data is not None:
        print(f"  Validation features: {val_path}")
    print(f"  Summary: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ CLUSTER 3 (Feature Engineering) COMPLETE!")
    print("📚 Ready to proceed to Cluster 4: Model Training")
    print("=" * 60)

if __name__ == "__main__":
    main()
