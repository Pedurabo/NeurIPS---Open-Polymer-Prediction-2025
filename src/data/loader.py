"""
Data loading utilities for NeurIPS Open Polymer Prediction 2025
Handles loading, validation, and basic exploration of competition data
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolymerDataLoader:
    """Main data loader class for polymer prediction competition"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.test_data = None
        self.sample_submission = None
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available data files
        
        Returns:
            Dictionary containing train, test, and sample submission data
        """
        logger.info("Loading all available data files...")
        
        data_files = {}
        
        # Load training data
        train_path = self.data_dir / "train.csv"
        if train_path.exists():
            self.train_data = pd.read_csv(train_path)
            data_files['train'] = self.train_data
            logger.info(f"Loaded training data: {self.train_data.shape}")
        else:
            logger.warning("Training data not found. Please ensure train.csv is in the data directory.")
            
        # Load test data
        test_path = self.data_dir / "test.csv"
        if test_path.exists():
            self.test_data = pd.read_csv(test_path)
            data_files['test'] = self.test_data
            logger.info(f"Loaded test data: {self.test_data.shape}")
        else:
            logger.warning("Test data not found. Please ensure test.csv is in the data directory.")
            
        # Load sample submission
        sample_path = self.data_dir / "sample_submission.csv"
        if sample_path.exists():
            self.sample_submission = pd.read_csv(sample_path)
            data_files['sample_submission'] = self.sample_submission
            logger.info(f"Loaded sample submission: {self.sample_submission.shape}")
        else:
            logger.warning("Sample submission not found.")
        
        # Load supplementary training datasets
        supplement_dir = self.data_dir / "train_supplement"
        if supplement_dir.exists():
            supplement_files = list(supplement_dir.glob("*.csv"))
            for file_path in supplement_files:
                try:
                    dataset_name = f"supplement_{file_path.stem}"
                    supplement_data = pd.read_csv(file_path)
                    data_files[dataset_name] = supplement_data
                    logger.info(f"Loaded {dataset_name}: {supplement_data.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path.name}: {e}")
        else:
            logger.info("No supplementary training data found.")
            
        return data_files
    
    def get_combined_training_data(self) -> pd.DataFrame:
        """
        Combine main training data with all supplementary datasets
        
        Returns:
            Combined training DataFrame
        """
        if self.train_data is None:
            logger.error("Main training data not loaded. Call load_all_data() first.")
            return pd.DataFrame()
        
        combined_data = [self.train_data]
        
        # Add supplementary datasets
        supplement_dir = self.data_dir / "train_supplement"
        if supplement_dir.exists():
            supplement_files = list(supplement_dir.glob("*.csv"))
            for file_path in supplement_files:
                try:
                    supplement_data = pd.read_csv(file_path)
                    # Check if columns are compatible
                    if set(supplement_data.columns) == set(self.train_data.columns):
                        combined_data.append(supplement_data)
                        logger.info(f"Added {file_path.stem}: {supplement_data.shape}")
                    else:
                        logger.warning(f"Skipping {file_path.stem}: incompatible columns")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path.name}: {e}")
        
        if len(combined_data) > 1:
            combined_df = pd.concat(combined_data, ignore_index=True)
            logger.info(f"Combined training data: {combined_df.shape}")
            return combined_df
        else:
            logger.info("No supplementary data to combine")
            return self.train_data
    
    def get_data_info(self) -> Dict[str, Dict]:
        """
        Get comprehensive information about loaded datasets
        
        Returns:
            Dictionary with dataset information
        """
        info = {}
        
        # Add main datasets
        for name, data in [('train', self.train_data), ('test', self.test_data), 
                          ('sample_submission', self.sample_submission)]:
            if data is not None:
                info[name] = {
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict(),
                    'memory_usage': data.memory_usage(deep=True).sum(),
                    'null_counts': data.isnull().sum().to_dict(),
                    'unique_counts': {col: data[col].nunique() for col in data.columns}
                }
        
        # Add supplementary datasets
        supplement_dir = self.data_dir / "train_supplement"
        if supplement_dir.exists():
            supplement_files = list(supplement_dir.glob("*.csv"))
            for file_path in supplement_files:
                try:
                    dataset_name = f"supplement_{file_path.stem}"
                    supplement_data = pd.read_csv(file_path)
                    info[dataset_name] = {
                        'shape': supplement_data.shape,
                        'columns': list(supplement_data.columns),
                        'dtypes': supplement_data.dtypes.to_dict(),
                        'memory_usage': supplement_data.memory_usage(deep=True).sum(),
                        'null_counts': supplement_data.isnull().sum().to_dict(),
                        'unique_counts': {col: supplement_data[col].nunique() for col in supplement_data.columns}
                    }
                except Exception as e:
                    logger.warning(f"Failed to get info for {file_path.name}: {e}")
                
        return info
    
    def validate_smiles(self, smiles_column: str = 'SMILES') -> Dict[str, int]:
        """
        Validate SMILES strings in the dataset
        
        Args:
            smiles_column: Name of the SMILES column
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_smiles': 0,
            'valid_smiles': 0,
            'invalid_smiles': 0,
            'empty_smiles': 0
        }
        
        if self.train_data is None:
            logger.error("Training data not loaded. Call load_all_data() first.")
            return validation_results
            
        if smiles_column not in self.train_data.columns:
            logger.error(f"Column '{smiles_column}' not found in training data.")
            return validation_results
            
        smiles_data = self.train_data[smiles_column]
        validation_results['total_smiles'] = len(smiles_data)
        
        # Check for empty/null SMILES
        empty_mask = smiles_data.isna() | (smiles_data == '') | (smiles_data == ' ')
        validation_results['empty_smiles'] = empty_mask.sum()
        
        # Basic SMILES validation (can be enhanced with RDKit)
        valid_smiles = 0
        invalid_smiles = 0
        
        for smiles in smiles_data.dropna():
            if self._is_valid_smiles_basic(str(smiles)):
                valid_smiles += 1
            else:
                invalid_smiles += 1
                
        validation_results['valid_smiles'] = valid_smiles
        validation_results['invalid_smiles'] = invalid_smiles
        
        return validation_results
    
    def _is_valid_smiles_basic(self, smiles: str) -> bool:
        """
        Basic SMILES validation (simple checks)
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if SMILES passes basic validation
        """
        if not isinstance(smiles, str):
            return False
            
        # Check for common SMILES characters
        valid_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]{}@+-=#$%:;,.\\/')
        
        # Check if string contains valid characters
        return all(c in valid_chars for c in smiles)
    
    def get_target_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics for target variables
        
        Returns:
            Dictionary with target variable statistics
        """
        if self.train_data is None:
            logger.error("Training data not loaded. Call load_all_data() first.")
            return {}
            
        target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        available_targets = [col for col in target_columns if col in self.train_data.columns]
        
        if not available_targets:
            logger.warning("No target columns found in training data.")
            return {}
            
        stats = {}
        for target in available_targets:
            target_data = self.train_data[target].dropna()
            stats[target] = {
                'count': len(target_data),
                'mean': target_data.mean(),
                'std': target_data.std(),
                'min': target_data.min(),
                'max': target_data.max(),
                'median': target_data.median(),
                'null_count': self.train_data[target].isnull().sum(),
                'unique_count': target_data.nunique()
            }
            
        return stats
    
    def get_sample_data(self, n_samples: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Get sample data for exploration
        
        Args:
            n_samples: Number of samples to return
            
        Returns:
            Dictionary with sample data
        """
        samples = {}
        
        if self.train_data is not None:
            samples['train'] = self.train_data.head(n_samples)
            
        if self.test_data is not None:
            samples['test'] = self.test_data.head(n_samples)
            
        if self.sample_submission is not None:
            samples['sample_submission'] = self.sample_submission.head(n_samples)
            
        return samples
    
    def check_data_quality(self) -> Dict[str, List[str]]:
        """
        Check for common data quality issues
        
        Returns:
            Dictionary with quality issues found
        """
        issues = {
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        if self.train_data is None:
            issues['errors'].append("Training data not loaded")
            return issues
            
        # Check for missing values
        null_counts = self.train_data.isnull().sum()
        high_null_cols = null_counts[null_counts > 0]
        
        if not high_null_cols.empty:
            issues['warnings'].append(f"Columns with missing values: {list(high_null_cols.index)}")
            
        # Check for duplicate rows
        duplicates = self.train_data.duplicated().sum()
        if duplicates > 0:
            issues['warnings'].append(f"Found {duplicates} duplicate rows")
            
        # Check for constant columns
        constant_cols = []
        for col in self.train_data.columns:
            if self.train_data[col].nunique() <= 1:
                constant_cols.append(col)
                
        if constant_cols:
            issues['warnings'].append(f"Constant columns: {constant_cols}")
            
        # Check data types
        object_cols = self.train_data.select_dtypes(include=['object']).columns
        if len(object_cols) > 1:  # Assuming SMILES is one object column
            issues['recommendations'].append(f"Consider converting object columns to appropriate types: {list(object_cols)}")
            
        return issues

def load_competition_data(data_dir: str = "data") -> PolymerDataLoader:
    """
    Convenience function to load competition data
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Configured PolymerDataLoader instance
    """
    loader = PolymerDataLoader(data_dir)
    loader.load_all_data()
    return loader
