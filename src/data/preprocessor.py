"""
Data preprocessing and cleaning for NeurIPS Open Polymer Prediction 2025
Handles missing values, SMILES validation, data quality, and preparation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolymerDataPreprocessor:
    """Comprehensive data preprocessing for polymer prediction data"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessor
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Target properties
        self.target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Store preprocessing statistics
        self.preprocessing_stats = {}
        self.imputation_strategies = {}
        
    def validate_polymer_smiles(self, smiles_series: pd.Series) -> Dict[str, int]:
        """
        Validate polymer SMILES strings (more sophisticated than basic validation)
        
        Args:
            smiles_series: Series containing SMILES strings
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_smiles': len(smiles_series),
            'valid_smiles': 0,
            'invalid_smiles': 0,
            'empty_smiles': 0,
            'polymer_smiles': 0,
            'monomer_smiles': 0
        }
        
        for smiles in smiles_series.dropna():
            if pd.isna(smiles) or str(smiles).strip() == '':
                validation_results['empty_smiles'] += 1
                continue
                
            smiles_str = str(smiles).strip()
            
            # Check for polymer-specific patterns
            if self._is_valid_polymer_smiles(smiles_str):
                validation_results['valid_smiles'] += 1
                validation_results['polymer_smiles'] += 1
            elif self._is_valid_monomer_smiles(smiles_str):
                validation_results['valid_smiles'] += 1
                validation_results['monomer_smiles'] += 1
            else:
                validation_results['invalid_smiles'] += 1
                
        return validation_results
    
    def _is_valid_polymer_smiles(self, smiles: str) -> bool:
        """
        Check if SMILES represents a valid polymer structure
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid polymer SMILES
        """
        # Polymer SMILES should contain:
        # 1. Connection points (*)
        # 2. Valid chemical elements
        # 3. Proper parentheses and brackets
        # 4. Reasonable length
        
        if len(smiles) < 5 or len(smiles) > 500:
            return False
            
        # Check for connection points (essential for polymers)
        if '*' not in smiles:
            return False
            
        # Check for valid chemical elements
        valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
        }
        
        # Extract potential element symbols
        import re
        element_pattern = r'[A-Z][a-z]?'
        elements = re.findall(element_pattern, smiles)
        
        # Check if all elements are valid
        for element in elements:
            if element not in valid_elements:
                return False
                
        # Check for balanced parentheses and brackets
        if not self._check_balanced_symbols(smiles):
            return False
            
        return True
    
    def _is_valid_monomer_smiles(self, smiles: str) -> bool:
        """
        Check if SMILES represents a valid monomer structure
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if valid monomer SMILES
        """
        # Monomer SMILES should be valid chemical structures
        # but may not have connection points
        
        if len(smiles) < 3 or len(smiles) > 200:
            return False
            
        # Check for valid chemical elements (same as polymer)
        valid_elements = {
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
        }
        
        import re
        element_pattern = r'[A-Z][a-z]?'
        elements = re.findall(element_pattern, smiles)
        
        for element in elements:
            if element not in valid_elements:
                return False
                
        # Check for balanced symbols
        if not self._check_balanced_symbols(smiles):
            return False
            
        return True
    
    def _check_balanced_symbols(self, smiles: str) -> bool:
        """
        Check if parentheses and brackets are balanced
        
        Args:
            smiles: SMILES string to check
            
        Returns:
            True if symbols are balanced
        """
        stack = []
        pairs = {')': '(', ']': '[', '}': '{'}
        
        for char in smiles:
            if char in '([{':
                stack.append(char)
            elif char in ')]}':
                if not stack or stack.pop() != pairs[char]:
                    return False
                    
        return len(stack) == 0
    
    def analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze patterns in missing values
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with missing value analysis
        """
        missing_analysis = {}
        
        # Overall missing statistics
        missing_counts = data[self.target_columns].isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        missing_analysis['overall'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'total_missing': missing_counts.sum(),
            'total_percentage': (missing_counts.sum() / (len(data) * len(self.target_columns))) * 100
        }
        
        # Pattern analysis
        missing_patterns = data[self.target_columns].isnull().value_counts()
        missing_analysis['patterns'] = {
            'unique_patterns': len(missing_patterns),
            'most_common_pattern': missing_patterns.index[0] if len(missing_patterns) > 0 else None,
            'most_common_count': missing_patterns.iloc[0] if len(missing_patterns) > 0 else 0
        }
        
        # Target-wise analysis
        target_analysis = {}
        for target in self.target_columns:
            target_data = data[target].dropna()
            if len(target_data) > 0:
                target_analysis[target] = {
                    'available_samples': len(target_data),
                    'missing_samples': missing_counts[target],
                    'coverage_percentage': (len(target_data) / len(data)) * 100,
                    'value_range': [target_data.min(), target_data.max()],
                    'mean': target_data.mean(),
                    'std': target_data.std()
                }
            else:
                target_analysis[target] = {
                    'available_samples': 0,
                    'missing_samples': len(data),
                    'coverage_percentage': 0.0,
                    'value_range': [None, None],
                    'mean': None,
                    'std': None
                }
                
        missing_analysis['targets'] = target_analysis
        
        return missing_analysis
    
    def create_imputation_strategies(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        Create appropriate imputation strategies for each target
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary mapping targets to imputation strategies
        """
        strategies = {}
        
        for target in self.target_columns:
            target_data = data[target].dropna()
            
            if len(target_data) == 0:
                # No data available - use default strategy
                strategies[target] = 'constant'
            elif len(target_data) < 100:
                # Very limited data - use simple strategies
                strategies[target] = 'median'
            elif len(target_data) < 1000:
                # Moderate data - use statistical strategies
                strategies[target] = 'knn'
            else:
                # Sufficient data - use advanced strategies
                strategies[target] = 'iterative'
                
        self.imputation_strategies = strategies
        return strategies
    
    def impute_missing_values(self, data: pd.DataFrame, 
                             strategy: str = 'auto') -> Tuple[pd.DataFrame, Dict]:
        """
        Impute missing values using specified strategies
        
        Args:
            data: DataFrame to impute
            strategy: Imputation strategy ('auto', 'simple', 'advanced')
            
        Returns:
            Tuple of (imputed_data, imputation_info)
        """
        imputed_data = data.copy()
        imputation_info = {}
        
        if strategy == 'auto':
            strategies = self.create_imputation_strategies(data)
        else:
            strategies = {target: strategy for target in self.target_columns}
            
        for target, target_strategy in strategies.items():
            logger.info(f"Imputing {target} using {target_strategy} strategy")
            
            if target_strategy == 'constant':
                # Use a reasonable default value based on domain knowledge
                default_values = {
                    'Tg': 100.0,      # Typical glass transition temperature
                    'FFV': 0.35,      # Typical fractional free volume
                    'Tc': 0.25,       # Typical thermal conductivity
                    'Density': 1.0,   # Typical polymer density
                    'Rg': 15.0        # Typical radius of gyration
                }
                imputed_data[target].fillna(default_values[target], inplace=True)
                imputation_info[target] = {
                    'strategy': 'constant',
                    'value': default_values[target],
                    'imputed_count': data[target].isnull().sum()
                }
                
            elif target_strategy == 'median':
                # Use median of available values
                median_val = data[target].median()
                imputed_data[target].fillna(median_val, inplace=True)
                imputation_info[target] = {
                    'strategy': 'median',
                    'value': median_val,
                    'imputed_count': data[target].isnull().sum()
                }
                
            elif target_strategy == 'knn':
                # Use KNN imputation (simplified version)
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                
                # Prepare data for KNN (only numerical columns)
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                numerical_data = data[numerical_cols].copy()
                
                # Impute the numerical data
                imputed_numerical = imputer.fit_transform(numerical_data)
                imputed_data[numerical_cols] = imputed_numerical
                
                imputation_info[target] = {
                    'strategy': 'knn',
                    'neighbors': 5,
                    'imputed_count': data[target].isnull().sum()
                }
                
            elif target_strategy == 'iterative':
                # Use iterative imputation (most sophisticated)
                try:
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer
                    enable_iterative_imputer
                    imputer = IterativeImputer(random_state=self.random_state, max_iter=10)
                except ImportError:
                    # Fallback to KNN if iterative imputer not available
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    target_strategy = 'knn'  # Update strategy for logging
                
                # Prepare data for imputation
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                numerical_data = data[numerical_cols].copy()
                
                # Impute the numerical data
                imputed_numerical = imputer.fit_transform(numerical_data)
                imputed_data[numerical_cols] = imputed_numerical
                
                imputation_info[target] = {
                    'strategy': target_strategy,
                    'max_iter': 10 if target_strategy == 'iterative' else None,
                    'neighbors': 5 if target_strategy == 'knn' else None,
                    'imputed_count': data[target].isnull().sum()
                }
                

                
        # Store imputation statistics
        self.preprocessing_stats['imputation'] = imputation_info
        
        return imputed_data, imputation_info
    
    def handle_outliers(self, data: pd.DataFrame, 
                       method: str = 'iqr', 
                       threshold: float = 1.5) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle outliers in target variables
        
        Args:
            data: DataFrame to process
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (cleaned_data, outlier_info)
        """
        cleaned_data = data.copy()
        outlier_info = {}
        
        for target in self.target_columns:
            if target not in data.columns:
                continue
                
            target_data = data[target].dropna()
            if len(target_data) == 0:
                continue
                
            if method == 'iqr':
                Q1 = target_data.quantile(0.25)
                Q3 = target_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((data[target] < lower_bound) | (data[target] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap outliers instead of removing them
                    cleaned_data.loc[data[target] < lower_bound, target] = lower_bound
                    cleaned_data.loc[data[target] > upper_bound, target] = upper_bound
                    
                    outlier_info[target] = {
                        'method': 'iqr',
                        'threshold': threshold,
                        'outliers_detected': outlier_count,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound,
                        'action': 'capped'
                    }
                    
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(target_data))
                outliers = z_scores > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Remove outliers
                    outlier_indices = target_data[outliers].index
                    cleaned_data.loc[outlier_indices, target] = np.nan
                    
                    outlier_info[target] = {
                        'method': 'zscore',
                        'threshold': threshold,
                        'outliers_detected': outlier_count,
                        'action': 'removed'
                    }
                    
        # Store outlier handling statistics
        self.preprocessing_stats['outliers'] = outlier_info
        
        return cleaned_data, outlier_info
    
    def create_train_validation_split(self, data: pd.DataFrame, 
                                    val_size: float = 0.2,
                                    stratify_by: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation split
        
        Args:
            data: DataFrame to split
            val_size: Fraction of data for validation
            stratify_by: Column to stratify by (if possible)
            
        Returns:
            Tuple of (train_data, val_data)
        """
        from sklearn.model_selection import train_test_split
        
        # Try to stratify by the target with best coverage
        if stratify_by is None:
            # Find target with best coverage for stratification
            target_coverage = {}
            for target in self.target_columns:
                if target in data.columns:
                    coverage = data[target].notna().sum() / len(data)
                    target_coverage[target] = coverage
                    
            if target_coverage:
                best_target = max(target_coverage, key=target_coverage.get)
                if target_coverage[best_target] > 0.5:  # Only stratify if >50% coverage
                    stratify_by = best_target
                    logger.info(f"Stratifying by {best_target} (coverage: {target_coverage[best_target]:.2%})")
                    
        if stratify_by and stratify_by in data.columns:
            # Create bins for stratification
            stratify_data = data[stratify_by].dropna()
            if len(stratify_data) > 0:
                # Create 5 bins for stratification
                bins = pd.cut(stratify_data, bins=5, labels=False)
                train_data, val_data = train_test_split(
                    data, 
                    test_size=val_size, 
                    random_state=self.random_state,
                    stratify=bins
                )
            else:
                # Fallback to random split
                train_data, val_data = train_test_split(
                    data, 
                    test_size=val_size, 
                    random_state=self.random_state
                )
        else:
            # Random split
            train_data, val_data = train_test_split(
                data, 
                test_size=val_size, 
                random_state=self.random_state
            )
            
        logger.info(f"Train/validation split: {len(train_data)}/{len(val_data)} samples")
        
        return train_data, val_data
    
    def standardize_features(self, data: pd.DataFrame, 
                           method: str = 'robust') -> Tuple[pd.DataFrame, Dict]:
        """
        Standardize numerical features
        
        Args:
            data: DataFrame to standardize
            method: Standardization method ('standard', 'robust', 'minmax')
            
        Returns:
            Tuple of (standardized_data, scaler_info)
        """
        from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
        
        standardized_data = data.copy()
        scaler_info = {}
        
        # Select numerical columns (exclude ID and SMILES)
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col != 'id']
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown standardization method: {method}")
            
        # Fit and transform numerical columns
        standardized_data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        
        scaler_info = {
            'method': method,
            'columns': numerical_cols,
            'scaler_type': type(scaler).__name__,
            'feature_names': numerical_cols
        }
        
        # Store scaler for later use
        self.preprocessing_stats['scaling'] = scaler_info
        
        return standardized_data, scaler_info
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get comprehensive preprocessing summary
        
        Returns:
            Dictionary with preprocessing statistics
        """
        summary = {
            'target_columns': self.target_columns,
            'imputation_strategies': self.imputation_strategies,
            'preprocessing_stats': self.preprocessing_stats,
            'random_state': self.random_state
        }
        
        return summary
    
    def save_preprocessed_data(self, data: pd.DataFrame, 
                             output_dir: str = "data/processed",
                             filename: str = "preprocessed_data.csv") -> str:
        """
        Save preprocessed data
        
        Args:
            data: Preprocessed DataFrame
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        data.to_csv(file_path, index=False)
        
        logger.info(f"Preprocessed data saved to: {file_path}")
        return str(file_path)

def preprocess_polymer_data(data: pd.DataFrame, 
                          random_state: int = 42,
                          imputation_strategy: str = 'auto',
                          outlier_method: str = 'iqr',
                          standardization: str = 'robust') -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for complete data preprocessing
    
    Args:
        data: Input DataFrame
        random_state: Random seed
        imputation_strategy: Strategy for missing values
        outlier_method: Method for outlier handling
        standardization: Method for feature standardization
        
    Returns:
        Tuple of (preprocessed_data, preprocessing_info)
    """
    preprocessor = PolymerDataPreprocessor(random_state=random_state)
    
    # Step 1: Analyze missing patterns
    missing_analysis = preprocessor.analyze_missing_patterns(data)
    logger.info(f"Missing value analysis completed")
    
    # Step 2: Impute missing values
    imputed_data, imputation_info = preprocessor.impute_missing_values(
        data, strategy=imputation_strategy
    )
    logger.info(f"Missing value imputation completed")
    
    # Step 3: Handle outliers
    cleaned_data, outlier_info = preprocessor.handle_outliers(
        imputed_data, method=outlier_method
    )
    logger.info(f"Outlier handling completed")
    
    # Step 4: Standardize features
    standardized_data, scaling_info = preprocessor.standardize_features(
        cleaned_data, method=standardization
    )
    logger.info(f"Feature standardization completed")
    
    # Get complete preprocessing summary
    preprocessing_summary = preprocessor.get_preprocessing_summary()
    preprocessing_summary['missing_analysis'] = missing_analysis
    preprocessing_summary['imputation_info'] = imputation_info
    preprocessing_summary['outlier_info'] = outlier_info
    preprocessing_summary['scaling_info'] = scaling_info
    
    return standardized_data, preprocessing_summary
