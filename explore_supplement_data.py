#!/usr/bin/env python3
"""
Explore supplementary training datasets to understand their structure
and see how to integrate them with the main training data
"""

import os
import sys
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.append('src')

from data.loader import PolymerDataLoader

def explore_supplement_data():
    """Explore the structure of supplementary datasets"""
    print("=" * 80)
    print("EXPLORING SUPPLEMENTARY TRAINING DATASETS")
    print("=" * 80)
    
    loader = PolymerDataLoader()
    
    # Load all data
    print("\n📊 Loading all available data...")
    data_files = loader.load_all_data()
    
    print(f"\n📁 Available datasets:")
    for name, data in data_files.items():
        print(f"  {name}: {data.shape}")
    
    # Explore main training data structure
    print(f"\n🔍 Main Training Data Structure:")
    if 'train' in data_files:
        train_data = data_files['train']
        print(f"  Shape: {train_data.shape}")
        print(f"  Columns: {list(train_data.columns)}")
        print(f"  Data types:")
        for col, dtype in train_data.dtypes.items():
            print(f"    {col}: {dtype}")
    
    # Explore supplementary datasets
    print(f"\n🔍 Supplementary Datasets Analysis:")
    supplement_datasets = {k: v for k, v in data_files.items() if k.startswith('supplement_')}
    
    for name, data in supplement_datasets.items():
        print(f"\n  📊 {name}:")
        print(f"    Shape: {data.shape}")
        print(f"    Columns: {list(data.columns)}")
        print(f"    Data types:")
        for col, dtype in data.dtypes.items():
            print(f"      {col}: {dtype}")
        
        # Check for missing values
        if data.isnull().sum().sum() > 0:
            print(f"    Missing values:")
            for col, missing in data.isnull().sum().items():
                if missing > 0:
                    pct = (missing / len(data)) * 100
                    print(f"      {col}: {missing:,} ({pct:.2f}%)")
        
        # Show sample data
        print(f"    Sample data (first 3 rows):")
        print(f"      {data.head(3).to_string()}")
        
        # Check for common columns with main training data
        if 'train' in data_files:
            main_cols = set(data_files['train'].columns)
            supp_cols = set(data.columns)
            common_cols = main_cols.intersection(supp_cols)
            print(f"    Common columns with main data: {list(common_cols)}")
            missing_cols = main_cols - supp_cols
            extra_cols = supp_cols - main_cols
            if missing_cols:
                print(f"    Missing columns: {list(missing_cols)}")
            if extra_cols:
                print(f"    Extra columns: {list(extra_cols)}")
    
    # Analyze potential integration strategies
    print(f"\n🎯 INTEGRATION STRATEGY ANALYSIS:")
    print("=" * 60)
    
    if 'train' in data_files and supplement_datasets:
        main_train = data_files['train']
        
        print(f"\n📊 Main Training Data:")
        print(f"  Total samples: {len(main_train):,}")
        print(f"  Target coverage:")
        target_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        for target in target_cols:
            if target in main_train.columns:
                coverage = main_train[target].notna().sum()
                pct = (coverage / len(main_train)) * 100
                print(f"    {target}: {coverage:,} samples ({pct:.2f}%)")
        
        print(f"\n🔄 Potential Integration Approaches:")
        
        # Approach 1: Direct column matching
        print(f"\n  1️⃣ Direct Column Matching:")
        for name, supp_data in supplement_datasets.items():
            if set(supp_data.columns) == set(main_train.columns):
                print(f"    ✅ {name}: Perfect column match - can be directly concatenated")
            else:
                common = set(supp_data.columns).intersection(set(main_train.columns))
                if len(common) >= 2:  # At least SMILES + one target
                    print(f"    🔄 {name}: Partial match - {len(common)} common columns")
                    print(f"      Common: {list(common)}")
                else:
                    print(f"    ❌ {name}: No useful column overlap")
        
        # Approach 2: Feature engineering from SMILES
        print(f"\n  2️⃣ SMILES-Based Feature Engineering:")
        for name, supp_data in supplement_datasets.items():
            if 'SMILES' in supp_data.columns:
                print(f"    🧪 {name}: Has SMILES - can extract molecular features")
                # Check what other columns exist
                other_cols = [col for col in supp_data.columns if col != 'SMILES']
                if other_cols:
                    print(f"      Additional data: {other_cols}")
            else:
                print(f"    ❌ {name}: No SMILES column")
        
        # Approach 3: Target property mapping
        print(f"\n  3️⃣ Target Property Mapping:")
        for name, supp_data in supplement_datasets.items():
            target_overlap = set(supp_data.columns).intersection(set(target_cols))
            if target_overlap:
                print(f"    🎯 {name}: Has target properties: {list(target_overlap)}")
                # Check coverage
                for target in target_overlap:
                    coverage = supp_data[target].notna().sum()
                    pct = (coverage / len(supp_data)) * 100
                    print(f"      {target}: {coverage:,} samples ({pct:.2f}%)")
            else:
                print(f"    ❌ {name}: No target properties found")
    
    print(f"\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    explore_supplement_data()
