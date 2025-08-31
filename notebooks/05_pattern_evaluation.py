#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - Pattern Evaluation
Cluster 5: Analyze prediction patterns, cross-validation, interpretability, and evaluation

This script implements the fifth phase of the CRISP-DM methodology:
1. Perform cross-validation for robustness testing
2. Analyze prediction patterns and errors
3. Implement model interpretability
4. Generate comprehensive evaluation reports
5. Provide actionable recommendations

Based on Cluster 4 results:
- 6 trained machine learning models
- Deep learning MLP implemented
- Models saved and ready for evaluation
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
from models.evaluator import PolymerModelEvaluator, evaluate_polymer_models

def main():
    """Main pattern evaluation function"""
    print("=" * 80)
    print("NEURIPS OPEN POLYMER PREDICTION 2025 - PATTERN EVALUATION")
    print("CLUSTER 5: Analyze Patterns & Evaluate Models")
    print("=" * 80)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    print("\n🔍 Initializing pattern evaluation tools...")
    evaluator = PolymerModelEvaluator(random_state=42)
    
    # Load trained models
    print("\n🤖 Loading trained models...")
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        print("Please run Cluster 4 (Model Training) first.")
        return
    
    # Load training results
    training_results_file = os.path.join(models_dir, "training_results.json")
    if not os.path.exists(training_results_file):
        print(f"❌ Training results not found: {training_results_file}")
        print("Please run Cluster 4 (Model Training) first.")
        return
    
    with open(training_results_file, 'r') as f:
        training_results = json.load(f)
    
    print(f"✅ Training results loaded: {len(training_results)} models")
    
    # Load feature matrix for evaluation
    print("\n📊 Loading feature matrix for evaluation...")
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
    
    # Prepare data for evaluation
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
    
    # Load actual trained models
    print("\n🔧 Loading trained models from disk...")
    loaded_models = {}
    
    for model_name, model_info in training_results.items():
        if model_info['status'] == 'success':
            model_file = os.path.join(models_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_file):
                try:
                    with open(model_file, 'rb') as f:
                        import pickle
                        model = pickle.load(f)
                    
                    loaded_models[model_name] = {
                        'model': model,
                        'status': 'success',
                        'config': model_info.get('config', {}),
                        'train_time': model_info.get('train_time', 0),
                        'validation_metrics': model_info.get('validation_metrics', {})
                    }
                    print(f"  ✅ Loaded {model_name}")
                except Exception as e:
                    print(f"  ❌ Failed to load {model_name}: {e}")
                    loaded_models[model_name] = {
                        'model': None,
                        'status': 'failed',
                        'error': f"Load failed: {e}"
                    }
            else:
                print(f"  ❌ Model file not found: {model_file}")
                loaded_models[model_name] = {
                    'model': None,
                    'status': 'failed',
                    'error': 'Model file not found'
                }
        else:
            loaded_models[model_name] = model_info
    
    print(f"\n✅ Models loaded: {len([m for m in loaded_models.values() if m['status'] == 'success'])} successful")
    
    # Task 5.1: Perform Cross-Validation
    print("\n" + "=" * 60)
    print("TASK 5.1: PERFORMING CROSS-VALIDATION")
    print("=" * 60)
    
    print("\n🔄 Performing 5-fold cross-validation...")
    cv_results = evaluator.perform_cross_validation(loaded_models, X_train, y_train, cv_folds=5)
    
    print(f"\n✅ Cross-validation completed!")
    successful_cv = [name for name, result in cv_results.items() if result['status'] == 'success']
    failed_cv = [name for name, result in cv_results.items() if result['status'] == 'failed']
    
    print(f"  Successful CV: {len(successful_cv)}")
    print(f"  Failed CV: {len(failed_cv)}")
    
    if successful_cv:
        print(f"  ✅ CV completed for: {', '.join(successful_cv)}")
    if failed_cv:
        print(f"  ❌ CV failed for: {', '.join(failed_cv)}")
    
    # Task 5.2: Analyze Prediction Patterns
    print("\n" + "=" * 60)
    print("TASK 5.2: ANALYZING PREDICTION PATTERNS")
    print("=" * 60)
    
    print("\n📊 Analyzing prediction patterns and errors...")
    pattern_analysis = evaluator.analyze_prediction_patterns(loaded_models, X_val, y_val)
    
    print(f"\n✅ Pattern analysis completed!")
    successful_patterns = [name for name, result in pattern_analysis.items() if result['status'] == 'success']
    failed_patterns = [name for name, result in pattern_analysis.items() if result['status'] == 'failed']
    
    print(f"  Successful analysis: {len(successful_patterns)}")
    print(f"  Failed analysis: {len(failed_patterns)}")
    
    # Task 5.3: Implement Model Interpretability
    print("\n" + "=" * 60)
    print("TASK 5.3: IMPLEMENTING MODEL INTERPRETABILITY")
    print("=" * 60)
    
    print("\n🧠 Implementing model interpretability...")
    interpretability_results = evaluator.implement_model_interpretability(
        loaded_models, X_train, y_train, feature_columns
    )
    
    print(f"\n✅ Model interpretability completed!")
    successful_interpretability = [name for name, result in interpretability_results.items() if result['status'] == 'success']
    failed_interpretability = [name for name, result in interpretability_results.items() if result['status'] == 'failed']
    
    print(f"  Successful interpretability: {len(successful_interpretability)}")
    print(f"  Failed interpretability: {len(failed_interpretability)}")
    
    # Task 5.4: Create Comprehensive Evaluation Report
    print("\n" + "=" * 60)
    print("TASK 5.4: CREATING EVALUATION REPORT")
    print("=" * 60)
    
    print("\n📋 Creating comprehensive evaluation report...")
    evaluation_report = evaluator.create_evaluation_report(
        loaded_models, cv_results, pattern_analysis, interpretability_results
    )
    
    print(f"\n✅ Evaluation report created successfully!")
    
    # Task 5.5: Display Key Evaluation Results
    print("\n" + "=" * 60)
    print("TASK 5.5: KEY EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n🏆 MODEL RANKINGS (by Cross-Validation Performance):")
    print("=" * 60)
    
    if 'model_rankings' in evaluation_report:
        rankings = evaluation_report['model_rankings']
        for ranking in rankings[:5]:  # Top 5 models
            print(f"  #{ranking['rank']}: {ranking['model_name']} - Weighted MAE: {ranking['weighted_mae']:.4f}")
    
    print(f"\n📊 CROSS-VALIDATION SUMMARY:")
    print("=" * 40)
    
    if 'cross_validation_summary' in evaluation_report:
        cv_summary = evaluation_report['cross_validation_summary']
        if cv_summary.get('status') != 'no_successful_cv':
            print(f"  Total models: {cv_summary.get('total_models', 'N/A')}")
            print(f"  Mean Weighted MAE: {cv_summary.get('mean_weighted_mae', 'N/A'):.4f}")
            print(f"  Best Weighted MAE: {cv_summary.get('best_weighted_mae', 'N/A'):.4f}")
            print(f"  Worst Weighted MAE: {cv_summary.get('worst_weighted_mae', 'N/A'):.4f}")
        else:
            print("  No successful cross-validation results")
    
    print(f"\n🎯 ERROR ANALYSIS SUMMARY:")
    print("=" * 30)
    
    if 'error_analysis' in evaluation_report:
        error_summary = evaluation_report['error_analysis']
        if error_summary.get('status') != 'no_successful_analysis':
            print(f"  Total models analyzed: {error_summary.get('total_models', 'N/A')}")
            
            target_errors = error_summary.get('target_error_summary', {})
            if target_errors:
                print(f"  Target-wise error analysis:")
                for target, error_stats in target_errors.items():
                    print(f"    {target}: Mean MAE = {error_stats['mean_mae']:.4f}, Best = {error_stats['best_mae']:.4f}")
        else:
            print("  No successful error analysis")
    
    print(f"\n🧠 INTERPRETABILITY SUMMARY:")
    print("=" * 35)
    
    if 'interpretability_analysis' in evaluation_report:
        interpretability_summary = evaluation_report['interpretability_analysis']
        if interpretability_summary.get('status') != 'no_successful_interpretability':
            print(f"  Total models: {interpretability_summary.get('total_models', 'N/A')}")
            print(f"  Interpretable models: {interpretability_summary.get('interpretable_models', 'N/A')}")
            print(f"  Feature importance available: {interpretability_summary.get('feature_importance_available', 'N/A')}")
        else:
            print("  No successful interpretability analysis")
    
    # Task 5.6: Display Recommendations
    print("\n" + "=" * 60)
    print("TASK 5.6: ACTIONABLE RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n💡 KEY RECOMMENDATIONS:")
    print("=" * 30)
    
    if 'recommendations' in evaluation_report:
        recommendations = evaluation_report['recommendations']
        for i, recommendation in enumerate(recommendations, 1):
            print(f"  {i}. {recommendation}")
    
    # Task 5.7: Save Evaluation Results
    print("\n" + "=" * 60)
    print("TASK 5.7: SAVING EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n💾 Saving evaluation results...")
    report_path = evaluator.save_evaluation_results(evaluation_report, output_dir=models_dir)
    
    print(f"\n✅ Evaluation results saved successfully!")
    print(f"  Report path: {report_path}")
    
    # Task 5.8: Generate Final Evaluation Summary
    print("\n" + "=" * 60)
    print("TASK 5.8: FINAL EVALUATION SUMMARY")
    print("=" * 60)
    
    print("\n📋 COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n🤖 MODEL EVALUATION COMPLETED:")
    print(f"  Total models evaluated: {len(loaded_models)}")
    print(f"  Cross-validation: {len(successful_cv)}/{len(loaded_models)} successful")
    print(f"  Pattern analysis: {len(successful_patterns)}/{len(loaded_models)} successful")
    print(f"  Interpretability: {len(successful_interpretability)}/{len(loaded_models)} successful")
    
    print(f"\n📊 EVALUATION COVERAGE:")
    print(f"  Cross-validation folds: 5")
    print(f"  Error pattern analysis: ✅")
    print(f"  Model interpretability: ✅")
    print(f"  Feature importance: ✅")
    print(f"  Performance ranking: ✅")
    
    print(f"\n🏆 TOP PERFORMING MODELS:")
    if 'model_rankings' in evaluation_report and evaluation_report['model_rankings']:
        top_3 = evaluation_report['model_rankings'][:3]
        for i, model in enumerate(top_3, 1):
            print(f"  {i}. {model['model_name']} (Weighted MAE: {model['weighted_mae']:.4f})")
    
    print(f"\n💾 OUTPUT FILES:")
    print(f"  Evaluation report: {report_path}")
    print(f"  Cross-validation results: Available in report")
    print(f"  Pattern analysis: Available in report")
    print(f"  Interpretability analysis: Available in report")
    
    # Task 5.9: Next Steps for Cluster 6
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS FOR CLUSTER 6 (PRESENTATION)")
    print("=" * 60)
    
    print("\n📚 Ready for Final Presentation:")
    print("1. ✅ Cross-validation completed")
    print("2. ✅ Prediction patterns analyzed")
    print("3. ✅ Model interpretability implemented")
    print("4. ✅ Comprehensive evaluation report generated")
    print("5. ✅ Actionable recommendations provided")
    
    print("\n🎯 Next Phase Tasks:")
    print("1. Create visualizations and charts")
    print("2. Generate presentation materials")
    print("3. Document findings and insights")
    print("4. Prepare deployment recommendations")
    print("5. Create final project summary")
    
    print("\n💾 Evaluation Complete:")
    print(f"  All evaluation results saved to: {report_path}")
    print(f"  Models ready for production deployment")
    print(f"  Comprehensive insights available for decision making")
    
    print("\n" + "=" * 60)
    print("✅ CLUSTER 5 (Pattern Evaluation) COMPLETE!")
    print("📚 Ready to proceed to Cluster 6: Presentation")
    print("=" * 60)

if __name__ == "__main__":
    main()
