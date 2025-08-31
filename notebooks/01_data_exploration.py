#!/usr/bin/env python3
"""
NeurIPS Open Polymer Prediction 2025 - Data Exploration
Cluster 1: Data Understanding & Exploration

This script implements the first phase of the CRISP-DM methodology:
1. Understanding Data - Load and examine competition data
2. Data Preparation - Initial data quality assessment
3. Data Selection - Identify relevant features and targets
4. Data Mining - Basic statistical analysis
5. Pattern Evaluation - Visual insights and correlations
6. Presentation - Comprehensive exploration report

Target Properties:
- Tg - Glass transition temperature
- FFV - Fractional free volume
- Tc - Thermal conductivity
- Density - Polymer density
- Rg - Radius of gyration
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('../src')

# Standard data science imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append('src')

from data.loader import PolymerDataLoader, load_competition_data
from utils.visualization import DataVisualizer, create_exploration_report

def main():
    """Main data exploration function"""
    print("=" * 80)
    print("NEURIPS OPEN POLYMER PREDICTION 2025 - DATA EXPLORATION")
    print("CLUSTER 1: Data Understanding & Exploration")
    print("=" * 80)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    print("\n🔍 Initializing data exploration tools...")
    loader = PolymerDataLoader()
    visualizer = DataVisualizer()
    
    # Check what data files are available
    print("\n📁 Checking available data files...")
    data_files = os.listdir('data')
    print(f"Files in data directory: {data_files}")
    
    # Load all available data
    print("\n📊 Loading competition data...")
    loaded_data = loader.load_all_data()
    print(f"Loaded datasets: {list(loaded_data.keys())}")
    
    if not loaded_data:
        print("\n⚠️ No data files found. Please ensure data is in the data/ directory.")
        print("Expected files: train.csv, test.csv, sample_submission.csv")
        return
    
    # Task 1.2: Analyze Target Variables
    print("\n🎯 Analyzing target variables...")
    target_stats = loader.get_target_statistics()
    
    if target_stats:
        print(f"\nFound {len(target_stats)} target variables:")
        for target, stats in target_stats.items():
            print(f"\n{target}:")
            print(f"  Count: {stats['count']:,}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Missing: {stats['null_count']:,} ({stats['null_count']/stats['count']*100:.2f}%)")
            print(f"  Unique values: {stats['unique_count']:,}")
    else:
        print("⚠️ No target variables found. Please ensure training data is loaded.")
    
    # Task 1.3: Explore SMILES
    print("\n🧪 Validating SMILES molecular representations...")
    smiles_validation = loader.validate_smiles()
    
    print("\nSMILES Validation Results:")
    for key, value in smiles_validation.items():
        print(f"  {key}: {value:,}")
    
    # Calculate validation percentages
    total = smiles_validation['total_smiles']
    if total > 0:
        valid_pct = (smiles_validation['valid_smiles'] / total) * 100
        invalid_pct = (smiles_validation['invalid_smiles'] / total) * 100
        empty_pct = (smiles_validation['empty_smiles'] / total) * 100
        
        print(f"\nValidation Percentages:")
        print(f"  Valid: {valid_pct:.2f}%")
        print(f"  Invalid: {invalid_pct:.2f}%")
        print(f"  Empty: {empty_pct:.2f}%")
        
        # Show sample SMILES if available
        if 'train' in loaded_data and loaded_data['train'] is not None:
            train_data = loaded_data['train']
            if 'SMILES' in train_data.columns:
                print(f"\n📝 Sample SMILES strings:")
                sample_smiles = train_data['SMILES'].dropna().head(5)
                for i, smiles in enumerate(sample_smiles, 1):
                    print(f"  {i}. {smiles}")
    
    # Task 1.4: Generate Statistics
    print("\n📊 Generating dataset statistics...")
    data_info = loader.get_data_info()
    
    print("\nDataset Overview:")
    for dataset_name, info in data_info.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"  Shape: {info['shape']}")
        print(f"  Memory: {info['memory_usage'] / 1024**2:.2f} MB")
        print(f"  Columns: {info['columns']}")
        
        # Show data types
        print(f"  Data Types:")
        for col, dtype in info['dtypes'].items():
            print(f"    {col}: {dtype}")
        
        # Show null counts
        null_counts = info['null_counts']
        if any(null_counts.values()):
            print(f"  Missing Values:")
            for col, count in null_counts.items():
                if count > 0:
                    pct = (count / info['shape'][0]) * 100
                    print(f"    {col}: {count:,} ({pct:.2f}%)")
        else:
            print(f"  Missing Values: None")
        
        # Show unique value counts
        print(f"  Unique Values:")
        for col, count in info['unique_counts'].items():
            print(f"    {col}: {count:,}")
    
    # Task 1.5: Data Quality Assessment
    print("\n🔍 Assessing data quality...")
    quality_issues = loader.check_data_quality()
    
    print("\nData Quality Assessment:")
    for issue_type, issues in quality_issues.items():
        if issues:
            print(f"\n{issue_type.upper()}:")
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print(f"\n✅ {issue_type.title()}: No issues found")
    
    # Additional data quality checks
    if 'train' in loaded_data and loaded_data['train'] is not None:
        train_data = loaded_data['train']
        
        print("\n🔍 Additional Quality Checks:")
        
        # Check for duplicate rows
        duplicates = train_data.duplicated().sum()
        print(f"  Duplicate rows: {duplicates:,}")
        
        # Check for constant columns
        constant_cols = []
        for col in train_data.columns:
            if train_data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"  Constant columns: {constant_cols}")
        else:
            print(f"  Constant columns: None")
        
        # Check for extreme outliers in numerical columns
        numerical_cols = train_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n  Numerical columns: {list(numerical_cols)}")
            
            # Check for potential outliers using IQR method
            outlier_summary = {}
            for col in numerical_cols:
                if col in target_stats:  # Only check target variables for now
                    Q1 = train_data[col].quantile(0.25)
                    Q3 = train_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((train_data[col] < lower_bound) | (train_data[col] > upper_bound)).sum()
                    outlier_summary[col] = outliers
            
            if outlier_summary:
                print(f"  Potential outliers (IQR method):")
                for col, count in outlier_summary.items():
                    if count > 0:
                        pct = (count / len(train_data)) * 100
                        print(f"    {col}: {count:,} ({pct:.2f}%)")
    
    # Create visualizations
    print("\n📈 Creating data visualizations...")
    
    # 1. Dataset overview
    if data_info:
        print("\n📊 Plotting dataset overview...")
        visualizer.plot_data_overview(data_info)
    
    # 2. Target variable distributions
    if target_stats:
        print("\n🎯 Plotting target variable distributions...")
        visualizer.plot_target_distributions(target_stats)
    
    # 3. Target variable correlations
    if 'train' in loaded_data and loaded_data['train'] is not None:
        print("\n🔗 Plotting target variable correlations...")
        visualizer.plot_target_correlations(loaded_data['train'])
        
        print("\n📊 Plotting missing values pattern...")
        visualizer.plot_missing_values_heatmap(loaded_data['train'])
        
        print("\n📈 Plotting feature distributions...")
        visualizer.plot_feature_distributions(loaded_data['train'])
    
    # 4. SMILES analysis
    if smiles_validation:
        print("\n🧪 Plotting SMILES validation analysis...")
        visualizer.plot_smiles_analysis(smiles_validation)
    
    # 5. Data quality report
    if quality_issues:
        print("\n📋 Plotting data quality report...")
        visualizer.plot_data_quality_report(quality_issues)
    
    # Generate comprehensive exploration report
    print("\n📋 Generating comprehensive exploration report...")
    create_exploration_report(loader, visualizer)
    
    # Summary of key findings
    print("\n🔍 KEY INSIGHTS FROM DATA EXPLORATION:")
    print("=" * 60)
    
    if target_stats:
        print("\n🎯 TARGET VARIABLES:")
        for target, stats in target_stats.items():
            print(f"  • {target}: {stats['count']:,} samples, range [{stats['min']:.4f}, {stats['max']:.4f}]")
            
        # Identify most challenging targets
        missing_targets = [(target, stats['null_count']) for target, stats in target_stats.items()]
        missing_targets.sort(key=lambda x: x[1], reverse=True)
        
        if missing_targets[0][1] > 0:
            print(f"\n⚠️ Most challenging target: {missing_targets[0][0]} ({missing_targets[0][1]:,} missing values)")
    
    if smiles_validation:
        print(f"\n🧪 SMILES DATA:")
        total = smiles_validation['total_smiles']
        valid_pct = (smiles_validation['valid_smiles'] / total) * 100 if total > 0 else 0
        print(f"  • Total SMILES: {total:,}")
        print(f"  • Valid SMILES: {valid_pct:.2f}%")
        
        if smiles_validation['invalid_smiles'] > 0:
            print(f"  • ⚠️ Invalid SMILES: {smiles_validation['invalid_smiles']:,} (needs attention)")
    
    if quality_issues:
        total_issues = sum(len(issues) for issues in quality_issues.values())
        print(f"\n📊 DATA QUALITY:")
        print(f"  • Total issues identified: {total_issues}")
        
        for issue_type, issues in quality_issues.items():
            if issues:
                print(f"    - {issue_type.title()}: {len(issues)} issues")
    
    print("\n" + "=" * 60)
    print("🚀 NEXT STEPS FOR CLUSTER 2 (Data Preparation):")
    print("=" * 60)
    print("1. Handle missing values in target variables")
    print("2. Clean and validate SMILES strings")
    print("3. Remove duplicate and invalid entries")
    print("4. Implement data standardization/normalization")
    print("5. Create train/validation splits")
    print("6. Prepare data for feature engineering")
    
    print("\n✅ Cluster 1 (Data Understanding) Complete!")
    print("📚 Ready to proceed to Cluster 2: Data Preparation & Cleaning")

if __name__ == "__main__":
    main()
