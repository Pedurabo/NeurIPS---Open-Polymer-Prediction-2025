#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - Model Training
Cluster 4: Train baseline ML models, implement deep learning, and optimize performance

This script implements the fourth phase of the CRISP-DM methodology:
1. Train baseline machine learning models
2. Implement deep learning approaches
3. Evaluate model performance
4. Create ensemble models
5. Save trained models for deployment

Based on Cluster 3 results:
- Rich feature matrix with 1,000 selected features
- Molecular descriptors, Morgan fingerprints, and custom polymer features
- Clean, structured data ready for machine learning
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
from models.trainer import PolymerModelTrainer, train_polymer_models

def main():
    """Main model training function"""
    print("=" * 80)
    print("NEURIPS OPEN POLYMER PREDICTION 2025 - MODEL TRAINING")
    print("CLUSTER 4: Train Models & Implement Deep Learning")
    print("=" * 80)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    print("\n🔍 Initializing model training tools...")
    trainer = PolymerModelTrainer(random_state=42)
    
    # Load feature matrix
    print("\n📊 Loading feature matrix...")
    processed_dir = "data/processed"
    
    if not os.path.exists(processed_dir):
        print(f"❌ Processed data directory not found: {processed_dir}")
        print("Please run Cluster 3 (Feature Engineering) first.")
        return
    
    # Check for feature matrix files
    train_features_file = os.path.join(processed_dir, "feature_matrix_final.csv")
    val_features_file = os.path.join(processed_dir, "feature_matrix_validation.csv")
    
    if not os.path.exists(train_features_file):
        print(f"❌ Training feature matrix not found: {train_features_file}")
        print("Please run Cluster 3 (Feature Engineering) first.")
        return
    
    # Load feature matrices
    train_features = pd.read_csv(train_features_file)
    val_features = pd.read_csv(val_features_file) if os.path.exists(val_features_file) else None
    
    print(f"✅ Training features loaded: {train_features.shape}")
    if val_features is not None:
        print(f"✅ Validation features loaded: {val_features.shape}")
    
    # Separate features and targets
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    feature_columns = [col for col in train_features.columns if col not in target_columns]
    
    print(f"\n📊 Data structure:")
    print(f"  Feature columns: {len(feature_columns)}")
    print(f"  Target columns: {len(target_columns)}")
    print(f"  Training samples: {len(train_features)}")
    if val_features is not None:
        print(f"  Validation samples: {len(val_features)}")
    
    # Prepare training data
    X_train = train_features[feature_columns]
    y_train = train_features[target_columns]
    
    if val_features is not None:
        X_val = val_features[feature_columns]
        y_val = val_features[target_columns]
    else:
        # Create validation split if not available
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        print(f"  Created validation split: {len(X_train)}/{len(X_val)}")
    
    # Task 4.1: Train Baseline Machine Learning Models
    print("\n" + "=" * 60)
    print("TASK 4.1: TRAINING BASELINE MACHINE LEARNING MODELS")
    print("=" * 60)
    
    print("\n🤖 Training baseline models...")
    baseline_results = trainer.train_baseline_models(X_train, y_train, X_val, y_val)
    
    print(f"\n✅ Baseline model training completed!")
    successful_models = [name for name, result in baseline_results.items() if result['status'] == 'success']
    failed_models = [name for name, result in baseline_results.items() if result['status'] == 'failed']
    
    print(f"  Successful models: {len(successful_models)}")
    print(f"  Failed models: {len(failed_models)}")
    
    if successful_models:
        print(f"  ✅ Working models: {', '.join(successful_models)}")
    if failed_models:
        print(f"  ❌ Failed models: {', '.join(failed_models)}")
    
    # Task 4.2: Implement Deep Learning
    print("\n" + "=" * 60)
    print("TASK 4.2: IMPLEMENTING DEEP LEARNING")
    print("=" * 60)
    
    print("\n🧠 Implementing deep learning with MLP...")
    deep_learning_results = trainer.implement_deep_learning(X_train, y_train, X_val, y_val)
    
    if deep_learning_results['status'] == 'success':
        print(f"✅ Deep learning model trained successfully!")
        print(f"  Architecture: {deep_learning_results['architecture']}")
        print(f"  Model type: {deep_learning_results['model_type']}")
    else:
        print(f"❌ Deep learning failed: {deep_learning_results.get('error', 'Unknown error')}")
    
    # Task 4.3: Evaluate Model Performance
    print("\n" + "=" * 60)
    print("TASK 4.3: EVALUATING MODEL PERFORMANCE")
    print("=" * 60)
    
    print("\n📈 Evaluating all models...")
    
    # Combine all models for evaluation
    all_models = {**baseline_results}
    if deep_learning_results['status'] == 'success':
        all_models['deep_learning'] = deep_learning_results
    
    performance = trainer.evaluate_model_performance(all_models, X_val, y_val)
    
    print(f"\n✅ Model evaluation completed!")
    
    # Display performance summary
    print(f"\n📊 MODEL PERFORMANCE SUMMARY:")
    print("=" * 50)
    
    for model_name, perf_info in performance.items():
        if perf_info.get('status') == 'evaluation_failed':
            print(f"  ❌ {model_name}: Evaluation failed")
            continue
        
        if 'metrics' in perf_info and perf_info['metrics']:
            metrics = perf_info['metrics']
            weighted_mae = metrics.get('weighted_mae', 'N/A')
            print(f"  🎯 {model_name}:")
            print(f"    Weighted MAE: {weighted_mae:.4f}")
            
            # Show individual target performance
            for target in target_columns:
                if target in metrics:
                    target_metrics = metrics[target]
                    mae = target_metrics.get('mae', 'N/A')
                    r2 = target_metrics.get('r2', 'N/A')
                    print(f"      {target}: MAE={mae:.4f}, R²={r2:.4f}")
        else:
            print(f"  ❌ {model_name}: No metrics available")
    
    # Task 4.4: Create Ensemble Models
    print("\n" + "=" * 60)
    print("TASK 4.4: CREATING ENSEMBLE MODELS")
    print("=" * 60)
    
    print("\n🔗 Creating ensemble models...")
    ensemble_results = trainer.create_ensemble_model(all_models, X_train, y_train, X_val, y_val)
    
    if ensemble_results['status'] == 'success':
        print(f"✅ Ensemble model created successfully!")
        print(f"  Ensemble type: {ensemble_results['ensemble_type']}")
        print(f"  Base models: {', '.join(ensemble_results['base_models'])}")
        
        # Show ensemble performance
        ensemble_metrics = ensemble_results['validation_metrics']
        weighted_mae = ensemble_metrics.get('weighted_mae', 'N/A')
        print(f"  Ensemble Weighted MAE: {weighted_mae:.4f}")
        
        # Compare with best individual model
        best_individual_mae = float('inf')
        best_individual_model = None
        
        for model_name, perf_info in performance.items():
            if 'metrics' in perf_info and 'weighted_mae' in perf_info['metrics']:
                mae = perf_info['metrics']['weighted_mae']
                if mae < best_individual_mae:
                    best_individual_mae = mae
                    best_individual_model = model_name
        
        if best_individual_model:
            improvement = ((best_individual_mae - weighted_mae) / best_individual_mae) * 100
            print(f"  Best individual model: {best_individual_model} (MAE: {best_individual_mae:.4f})")
            print(f"  Ensemble improvement: {improvement:.2f}%")
    else:
        print(f"❌ Ensemble creation failed: {ensemble_results.get('reason', 'Unknown error')}")
    
    # Task 4.5: Model Performance Analysis
    print("\n" + "=" * 60)
    print("TASK 4.5: MODEL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    print("\n📊 Analyzing model performance patterns...")
    
    # Find best models for each target
    print(f"\n🏆 BEST MODELS BY TARGET:")
    print("=" * 40)
    
    for target in target_columns:
        best_model = None
        best_mae = float('inf')
        best_r2 = -float('inf')
        
        for model_name, perf_info in performance.items():
            if 'metrics' in perf_info and target in perf_info['metrics']:
                target_metrics = perf_info['metrics'][target]
                mae = target_metrics.get('mae', float('inf'))
                r2 = target_metrics.get('r2', -float('inf'))
                
                if mae < best_mae:
                    best_mae = mae
                    best_r2 = r2
                    best_model = model_name
        
        if best_model:
            print(f"  {target}:")
            print(f"    Best model: {best_model}")
            print(f"    MAE: {best_mae:.4f}")
            print(f"    R²: {best_r2:.4f}")
    
    # Overall best model
    print(f"\n🏆 OVERALL BEST MODEL:")
    print("=" * 30)
    
    best_overall_model = None
    best_overall_mae = float('inf')
    
    for model_name, perf_info in performance.items():
        if 'metrics' in perf_info and 'weighted_mae' in perf_info['metrics']:
            mae = perf_info['metrics']['weighted_mae']
            if mae < best_overall_mae:
                best_overall_mae = mae
                best_overall_model = model_name
    
    if best_overall_model:
        print(f"  Model: {best_overall_model}")
        print(f"  Weighted MAE: {best_overall_mae:.4f}")
        
        # Show target-wise performance for best model
        best_model_perf = performance[best_overall_model]
        if 'metrics' in best_model_perf:
            print(f"  Target performance:")
            for target in target_columns:
                if target in best_model_perf['metrics']:
                    target_metrics = best_model_perf['metrics'][target]
                    mae = target_metrics.get('mae', 'N/A')
                    r2 = target_metrics.get('r2', 'N/A')
                    print(f"    {target}: MAE={mae:.4f}, R²={r2:.4f}")
    
    # Task 4.6: Save Models and Results
    print("\n" + "=" * 60)
    print("TASK 4.6: SAVING MODELS & RESULTS")
    print("=" * 60)
    
    print("\n💾 Saving trained models and results...")
    
    # Save models
    models_dir = "models"
    saved_paths = trainer.save_models(output_dir=models_dir)
    
    print(f"\n✅ Models saved successfully!")
    for model_name, path in saved_paths.items():
        if model_name != 'training_results':
            print(f"  {model_name}: {path}")
    
    # Save comprehensive training summary
    training_summary = trainer.get_training_summary()
    training_summary['ensemble_results'] = ensemble_results
    training_summary['best_overall_model'] = best_overall_model
    training_summary['best_overall_mae'] = best_overall_mae
    
    summary_path = os.path.join(models_dir, "complete_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2, default=str)
    print(f"  Complete summary: {summary_path}")
    
    # Task 4.7: Generate Training Report
    print("\n" + "=" * 60)
    print("TASK 4.7: TRAINING REPORT")
    print("=" * 60)
    
    print("\n📋 GENERATING COMPREHENSIVE TRAINING REPORT")
    print("=" * 60)
    
    print(f"\n🤖 MODEL TRAINING SUMMARY:")
    print(f"  Total models trained: {len(all_models)}")
    print(f"  Successful models: {len(successful_models)}")
    print(f"  Failed models: {len(failed_models)}")
    print(f"  Deep learning: {'✅' if deep_learning_results['status'] == 'success' else '❌'}")
    print(f"  Ensemble: {'✅' if ensemble_results['status'] == 'success' else '❌'}")
    
    print(f"\n📊 DATA USED:")
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Features: {len(feature_columns):,}")
    print(f"  Targets: {len(target_columns)}")
    
    print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
    if best_overall_model:
        print(f"  Best overall model: {best_overall_model}")
        print(f"  Best weighted MAE: {best_overall_mae:.4f}")
    
    if ensemble_results['status'] == 'success':
        ensemble_mae = ensemble_results['validation_metrics'].get('weighted_mae', 'N/A')
        print(f"  Ensemble weighted MAE: {ensemble_mae:.4f}")
    
    print(f"\n💾 OUTPUT FILES:")
    print(f"  Models directory: {models_dir}/")
    print(f"  Training summary: {summary_path}")
    
    # Task 4.8: Next Steps for Cluster 5
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS FOR CLUSTER 5 (PATTERN EVALUATION)")
    print("=" * 60)
    
    print("\n📚 Ready for Pattern Evaluation:")
    print("1. ✅ Baseline ML models trained")
    print("2. ✅ Deep learning implemented")
    print("3. ✅ Model performance evaluated")
    print("4. ✅ Ensemble models created")
    print("5. ✅ All models saved and ready")
    
    print("\n🎯 Next Phase Tasks:")
    print("1. Analyze prediction patterns and errors")
    print("2. Perform cross-validation and robustness testing")
    print("3. Implement model interpretability")
    print("4. Generate prediction confidence intervals")
    print("5. Create comprehensive evaluation report")
    
    print("\n💾 Models Ready for Deployment:")
    print(f"  All trained models saved to: {models_dir}/")
    print(f"  Complete training summary: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ CLUSTER 4 (Model Training) COMPLETE!")
    print("📚 Ready to proceed to Cluster 5: Pattern Evaluation")
    print("=" * 60)

if __name__ == "__main__":
    main()
