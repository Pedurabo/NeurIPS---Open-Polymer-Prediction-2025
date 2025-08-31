#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - Data Preparation
Cluster 2: Data Preparation & Cleaning

This script implements the second phase of the CRISP-DM methodology:
1. Handle missing values in target variables
2. Clean and validate SMILES strings
3. Remove duplicate and invalid entries
4. Implement data standardization/normalization
5. Create train/validation splits
6. Prepare data for feature engineering

Based on Cluster 1 findings:
- Massive missing values (90%+ in most targets)
- SMILES validation needs improvement
- Data quality issues to address
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

# Import our custom modules
from data.loader import PolymerDataLoader
from data.preprocessor import PolymerDataPreprocessor, preprocess_polymer_data
from utils.visualization import DataVisualizer
from data.supplement_integrator import integrate_all_supplementary_data

def main():
    """Main data preparation function"""
    print("=" * 80)
    print("NEURIPS OPEN POLYMER PREDICTION 2025 - DATA PREPARATION")
    print("CLUSTER 2: Data Preparation & Cleaning")
    print("=" * 80)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    print("\n🔍 Initializing data preparation tools...")
    loader = PolymerDataLoader()
    preprocessor = PolymerDataPreprocessor(random_state=42)
    visualizer = DataVisualizer()
    
    # Load data
    print("\n📊 Loading competition data...")
    data_files = loader.load_all_data()
    
    if not data_files or 'train' not in data_files:
        print("❌ Training data not found. Please ensure data is loaded.")
        return
    
    # Get combined training data (main + supplementary)
    print("\n🔄 Combining main training data with supplementary datasets...")
    train_data = loader.get_combined_training_data()
    
    if train_data.empty:
        print("❌ No training data available.")
        return
        
    print(f"✅ Combined training data loaded: {train_data.shape}")
    
    # Show breakdown of datasets
    print(f"\n📊 Dataset Breakdown:")
    print(f"  Main training data: {data_files['train'].shape}")
    supplement_count = len([k for k in data_files.keys() if k.startswith('supplement_')])
    if supplement_count > 0:
        print(f"  Supplementary datasets: {supplement_count}")
        for key in data_files.keys():
            if key.startswith('supplement_'):
                print(f"    {key}: {data_files[key].shape}")
    print(f"  Total combined: {train_data.shape}")
    
    # Task 2.0: Integrate Supplementary Data
    print("\n" + "=" * 60)
    print("TASK 2.0: SUPPLEMENTARY DATA INTEGRATION")
    print("=" * 60)
    
    print("\n🔄 Integrating supplementary datasets to improve target coverage...")
    
    # Get supplementary datasets
    supplement_datasets = {k: v for k, v in data_files.items() if k.startswith('supplement_')}
    
    if supplement_datasets:
        print(f"Found {len(supplement_datasets)} supplementary datasets:")
        for name, data in supplement_datasets.items():
            print(f"  {name}: {data.shape}")
        
        # Integrate supplementary data
        integrated_data, integration_summary = integrate_all_supplementary_data(
            train_data, supplement_datasets
        )
        
        print(f"\n✅ Supplementary data integration completed!")
        print(f"  Original samples: {len(train_data):,}")
        print(f"  Final samples: {len(integrated_data):,}")
        print(f"  Target improvements:")
        
        for target, stats in integration_summary['target_coverage_analysis'].items():
            if stats['improvement'] > 0:
                print(f"    {target}: +{stats['improvement']:,} samples "
                      f"({stats['improvement_percentage']:.1f}% improvement)")
        
        # Use integrated data for further processing
        train_data = integrated_data
        print(f"\n📊 Using integrated data: {train_data.shape}")
    else:
        print("No supplementary datasets found.")
    
    # Task 2.1: Handle Missing Values in Target Variables
    print("\n" + "=" * 60)
    print("TASK 2.1: HANDLING MISSING VALUES")
    print("=" * 60)
    
    # Analyze missing patterns
    print("\n🔍 Analyzing missing value patterns...")
    missing_analysis = preprocessor.analyze_missing_patterns(train_data)
    
    print("\nMissing Value Analysis:")
    print(f"  Total missing values: {missing_analysis['overall']['total_missing']:,}")
    print(f"  Overall missing percentage: {missing_analysis['overall']['total_percentage']:.2f}%")
    
    print("\nTarget-wise missing values:")
    for target, stats in missing_analysis['targets'].items():
        print(f"  {target}: {stats['missing_samples']:,} missing ({stats['coverage_percentage']:.2f}% coverage)")
    
    # Create imputation strategies
    print("\n🎯 Creating imputation strategies...")
    strategies = preprocessor.create_imputation_strategies(train_data)
    
    print("\nImputation Strategies:")
    for target, strategy in strategies.items():
        print(f"  {target}: {strategy}")
    
    # Impute missing values
    print("\n🔄 Imputing missing values...")
    imputed_data, imputation_info = preprocessor.impute_missing_values(
        train_data, strategy='auto'
    )
    
    print("\nImputation Results:")
    for target, info in imputation_info.items():
        print(f"  {target}: {info['imputed_count']:,} values imputed using {info['strategy']} strategy")
    
    # Task 2.2: Clean and Validate SMILES Strings
    print("\n" + "=" * 60)
    print("TASK 2.2: SMILES VALIDATION & CLEANING")
    print("=" * 60)
    
    # Validate SMILES using improved validation
    print("\n🧪 Validating SMILES with polymer-specific rules...")
    smiles_validation = preprocessor.validate_polymer_smiles(train_data['SMILES'])
    
    print("\nSMILES Validation Results:")
    for key, value in smiles_validation.items():
        print(f"  {key}: {value:,}")
    
    # Calculate validation percentages
    total = smiles_validation['total_smiles']
    if total > 0:
        valid_pct = (smiles_validation['valid_smiles'] / total) * 100
        polymer_pct = (smiles_validation['polymer_smiles'] / total) * 100
        monomer_pct = (smiles_validation['monomer_smiles'] / total) * 100
        
        print(f"\nValidation Percentages:")
        print(f"  Valid SMILES: {valid_pct:.2f}%")
        print(f"  Polymer SMILES: {polymer_pct:.2f}%")
        print(f"  Monomer SMILES: {monomer_pct:.2f}%")
    
    # Task 2.3: Remove Duplicate and Invalid Entries
    print("\n" + "=" * 60)
    print("TASK 2.3: DATA CLEANING")
    print("=" * 60)
    
    # Check for duplicates
    print("\n🔍 Checking for duplicate entries...")
    initial_rows = len(imputed_data)
    duplicates = imputed_data.duplicated().sum()
    
    if duplicates > 0:
        print(f"  Found {duplicates:,} duplicate rows")
        imputed_data = imputed_data.drop_duplicates()
        print(f"  Removed duplicates: {len(imputed_data):,} rows remaining")
    else:
        print("  ✅ No duplicate rows found")
    
    # Check for constant columns
    print("\n🔍 Checking for constant columns...")
    constant_cols = []
    for col in imputed_data.columns:
        if imputed_data[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"  Found constant columns: {constant_cols}")
        imputed_data = imputed_data.drop(columns=constant_cols)
        print(f"  Removed constant columns: {len(imputed_data.columns)} columns remaining")
    else:
        print("  ✅ No constant columns found")
    
    # Task 2.4: Handle Outliers
    print("\n" + "=" * 60)
    print("TASK 2.4: OUTLIER HANDLING")
    print("=" * 60)
    
    print("\n🔍 Handling outliers using IQR method...")
    cleaned_data, outlier_info = preprocessor.handle_outliers(
        imputed_data, method='iqr', threshold=1.5
    )
    
    print("\nOutlier Handling Results:")
    for target, info in outlier_info.items():
        if info['outliers_detected'] > 0:
            print(f"  {target}: {info['outliers_detected']:,} outliers {info['action']}")
            print(f"    Range: [{info['lower_bound']:.4f}, {info['upper_bound']:.4f}]")
    
    # Task 2.5: Data Standardization
    print("\n" + "=" * 60)
    print("TASK 2.5: FEATURE STANDARDIZATION")
    print("=" * 60)
    
    print("\n📊 Standardizing numerical features...")
    standardized_data, scaling_info = preprocessor.standardize_features(
        cleaned_data, method='robust'
    )
    
    print(f"\nStandardization completed:")
    print(f"  Method: {scaling_info['method']}")
    print(f"  Scaler: {scaling_info['scaler_type']}")
    print(f"  Features standardized: {len(scaling_info['columns'])}")
    
    # Task 2.6: Create Train/Validation Split
    print("\n" + "=" * 60)
    print("TASK 2.6: TRAIN/VALIDATION SPLIT")
    print("=" * 60)
    
    print("\n✂️ Creating train/validation split...")
    train_data_final, val_data = preprocessor.create_train_validation_split(
        standardized_data, val_size=0.2
    )
    
    print(f"\nSplit completed:")
    print(f"  Training samples: {len(train_data_final):,}")
    print(f"  Validation samples: {len(val_data):,}")
    print(f"  Split ratio: {len(train_data_final)}/{len(val_data)} ({len(train_data_final)/len(standardized_data)*100:.1f}%/{len(val_data)/len(standardized_data)*100:.1f}%)")
    
    # Task 2.7: Data Quality Assessment After Preprocessing
    print("\n" + "=" * 60)
    print("TASK 2.7: POST-PREPROCESSING QUALITY ASSESSMENT")
    print("=" * 60)
    
    # Check final data quality
    print("\n🔍 Final data quality check...")
    
    # Missing values after preprocessing
    target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    final_missing = train_data_final[target_columns].isnull().sum()
    print(f"\nMissing values after preprocessing:")
    for target in target_columns:
        missing_count = final_missing[target]
        missing_pct = (missing_count / len(train_data_final)) * 100
        print(f"  {target}: {missing_count:,} ({missing_pct:.2f}%)")
    
    # Data types and ranges
    print(f"\nFinal data types:")
    for col in train_data_final.columns:
        dtype = train_data_final[col].dtype
        if np.issubdtype(dtype, np.number):
            col_data = train_data_final[col].dropna()
            if len(col_data) > 0:
                print(f"  {col}: {dtype} - Range: [{col_data.min():.4f}, {col_data.max():.4f}]")
            else:
                print(f"  {col}: {dtype} - No data")
        else:
            print(f"  {col}: {dtype}")
    
    # Task 2.8: Save Preprocessed Data
    print("\n" + "=" * 60)
    print("TASK 2.8: SAVING PREPROCESSED DATA")
    print("=" * 60)
    
    # Create processed data directory
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save preprocessed data
    print("\n💾 Saving preprocessed data...")
    
    # Save training data
    train_path = os.path.join(processed_dir, "train_preprocessed.csv")
    train_data_final.to_csv(train_path, index=False)
    print(f"  ✅ Training data saved: {train_path}")
    
    # Save validation data
    val_path = os.path.join(processed_dir, "val_preprocessed.csv")
    val_data.to_csv(val_path, index=False)
    print(f"  ✅ Validation data saved: {val_path}")
    
    # Save preprocessing summary
    preprocessing_summary = preprocessor.get_preprocessing_summary()
    preprocessing_summary['missing_analysis'] = missing_analysis
    preprocessing_summary['missing_analysis'] = missing_analysis
    preprocessing_summary['imputation_info'] = imputation_info
    preprocessing_summary['outlier_info'] = outlier_info
    preprocessing_summary['scaling_info'] = scaling_info
    preprocessing_summary['final_shapes'] = {
        'original': train_data.shape,
        'final_train': train_data_final.shape,
        'final_val': val_data.shape
    }
    
    import json
    summary_path = os.path.join(processed_dir, "preprocessing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(preprocessing_summary, f, indent=2, default=str)
    print(f"  ✅ Preprocessing summary saved: {summary_path}")
    
    # Task 2.9: Generate Preprocessing Report
    print("\n" + "=" * 60)
    print("TASK 2.9: PREPROCESSING REPORT")
    print("=" * 60)
    
    print("\n📋 GENERATING PREPROCESSING REPORT")
    print("=" * 60)
    
    print(f"\n📊 DATA TRANSFORMATION SUMMARY:")
    print(f"  Original shape: {train_data.shape}")
    print(f"  Final training shape: {train_data_final.shape}")
    print(f"  Final validation shape: {val_data.shape}")
    print(f"  Total samples processed: {len(standardized_data):,}")
    
    print(f"\n🎯 TARGET VARIABLE STATUS:")
    for target in target_columns:
        if target in train_data_final.columns:
            target_data = train_data_final[target].dropna()
            coverage = len(target_data) / len(train_data_final) * 100
            print(f"  {target}: {len(target_data):,} samples ({coverage:.2f}% coverage)")
    
    print(f"\n🧪 SMILES VALIDATION:")
    print(f"  Total SMILES: {smiles_validation['total_smiles']:,}")
    print(f"  Valid SMILES: {smiles_validation['valid_smiles']:,} ({smiles_validation['valid_smiles']/smiles_validation['total_smiles']*100:.2f}%)")
    print(f"  Polymer SMILES: {smiles_validation['polymer_smiles']:,}")
    print(f"  Monomer SMILES: {smiles_validation['monomer_smiles']:,}")
    
    print(f"\n🔄 IMPUTATION STRATEGIES:")
    for target, strategy in strategies.items():
        print(f"  {target}: {strategy}")
    
    print(f"\n📈 OUTLIER HANDLING:")
    total_outliers = sum(info['outliers_detected'] for info in outlier_info.values())
    if total_outliers > 0:
        print(f"  Total outliers handled: {total_outliers:,}")
        for target, info in outlier_info.items():
            if info['outliers_detected'] > 0:
                print(f"    {target}: {info['outliers_detected']:,} outliers {info['action']}")
    else:
        print(f"  ✅ No outliers detected")
    
    print(f"\n📊 FEATURE STANDARDIZATION:")
    print(f"  Method: {scaling_info['method']}")
    print(f"  Scaler: {scaling_info['scaler_type']}")
    print(f"  Features standardized: {len(scaling_info['columns'])}")
    
    print(f"\n✂️ DATA SPLITTING:")
    print(f"  Training samples: {len(train_data_final):,}")
    print(f"  Validation samples: {len(val_data):,}")
    print(f"  Split ratio: {len(train_data_final)}/{len(val_data)}")
    
    # Task 2.10: Next Steps for Cluster 3
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS FOR CLUSTER 3 (FEATURE ENGINEERING)")
    print("=" * 60)
    
    print("\n📚 Ready for Feature Engineering:")
    print("1. ✅ Data cleaned and validated")
    print("2. ✅ Missing values handled")
    print("3. ✅ Outliers processed")
    print("4. ✅ Features standardized")
    print("5. ✅ Train/validation split created")
    
    print("\n🎯 Next Phase Tasks:")
    print("1. Extract molecular descriptors from SMILES")
    print("2. Generate Morgan fingerprints")
    print("3. Create custom polymer-specific features")
    print("4. Implement feature selection")
    print("5. Prepare final feature matrix")
    
    print("\n💾 Data Files Ready:")
    print(f"  Training: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Summary: {summary_path}")
    
    print("\n" + "=" * 60)
    print("✅ CLUSTER 2 (Data Preparation) COMPLETE!")
    print("📚 Ready to proceed to Cluster 3: Feature Engineering")
    print("=" * 60)

if __name__ == "__main__":
    main()
