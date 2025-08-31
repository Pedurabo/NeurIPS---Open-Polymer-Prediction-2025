"""
Feature Engineering for NeurIPS Open Polymer Prediction 2025
Cluster 3: Extract molecular descriptors, generate fingerprints, and create features
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

class PolymerFeatureEngineer:
    """Comprehensive feature engineering for polymer prediction data"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the feature engineer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Target properties
        self.target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
        # Store feature engineering statistics
        self.feature_stats = {}
        self.feature_names = []
        self.selected_features = []
        
    def extract_molecular_descriptors(self, smiles_series: pd.Series) -> pd.DataFrame:
        """
        Extract molecular descriptors from SMILES strings
        
        Args:
            smiles_series: Series containing SMILES strings
            
        Returns:
            DataFrame with molecular descriptors
        """
        logger.info("Extracting molecular descriptors from SMILES...")
        
        descriptors_df = pd.DataFrame(index=smiles_series.index)
        
        # Basic SMILES-based features
        descriptors_df['smiles_length'] = smiles_series.str.len()
        descriptors_df['connection_points'] = smiles_series.str.count('\*')
        descriptors_df['parentheses_count'] = smiles_series.str.count('\(') + smiles_series.str.count('\)')
        descriptors_df['brackets_count'] = smiles_series.str.count('\[') + smiles_series.str.count('\]')
        descriptors_df['braces_count'] = smiles_series.str.count('\{') + smiles_series.str.count('\}')
        
        # Element counts
        descriptors_df['carbon_count'] = smiles_series.str.count('C')
        descriptors_df['nitrogen_count'] = smiles_series.str.count('N')
        descriptors_df['oxygen_count'] = smiles_series.str.count('O')
        descriptors_df['hydrogen_count'] = smiles_series.str.count('H')
        descriptors_df['fluorine_count'] = smiles_series.str.count('F')
        descriptors_df['chlorine_count'] = smiles_series.str.count('Cl')
        descriptors_df['bromine_count'] = smiles_series.str.count('Br')
        descriptors_df['iodine_count'] = smiles_series.str.count('I')
        descriptors_df['sulfur_count'] = smiles_series.str.count('S')
        descriptors_df['phosphorus_count'] = smiles_series.str.count('P')
        descriptors_df['silicon_count'] = smiles_series.str.count('Si')
        
        # Bond counts
        descriptors_df['single_bonds'] = smiles_series.str.count('-')
        descriptors_df['double_bonds'] = smiles_series.str.count('=')
        descriptors_df['triple_bonds'] = smiles_series.str.count('#')
        
        # Ring features
        descriptors_df['ring_count'] = smiles_series.str.count('\d')
        
        # Polymer-specific features
        descriptors_df['polymer_indicator'] = (descriptors_df['connection_points'] > 0).astype(int)
        descriptors_df['complexity_score'] = (descriptors_df['smiles_length'] * 
                                           descriptors_df['parentheses_count'] * 
                                           descriptors_df['brackets_count'])
        
        # Molecular weight approximation (rough estimate)
        descriptors_df['molecular_weight'] = (
            descriptors_df['carbon_count'] * 12.01 +
            descriptors_df['nitrogen_count'] * 14.01 +
            descriptors_df['oxygen_count'] * 16.00 +
            descriptors_df['hydrogen_count'] * 1.01 +
            descriptors_df['fluorine_count'] * 19.00 +
            descriptors_df['chlorine_count'] * 35.45 +
            descriptors_df['bromine_count'] * 79.90 +
            descriptors_df['iodine_count'] * 126.90 +
            descriptors_df['sulfur_count'] * 32.07 +
            descriptors_df['phosphorus_count'] * 30.97 +
            descriptors_df['silicon_count'] * 28.09
        )
        
        # Fill NaN values with 0 for count-based features
        count_columns = [col for col in descriptors_df.columns if 'count' in col]
        descriptors_df[count_columns] = descriptors_df[count_columns].fillna(0)
        
        # Fill other NaN values with median
        descriptors_df = descriptors_df.fillna(descriptors_df.median())
        
        logger.info(f"Extracted {len(descriptors_df.columns)} molecular descriptors")
        return descriptors_df
    
    def generate_morgan_fingerprints(self, smiles_series: pd.Series, 
                                   radius: int = 2, 
                                   n_bits: int = 2048) -> pd.DataFrame:
        """
        Generate Morgan fingerprints from SMILES strings
        
        Args:
            smiles_series: Series containing SMILES strings
            radius: Morgan fingerprint radius
            n_bits: Number of bits in fingerprint
            
        Returns:
            DataFrame with Morgan fingerprints
        """
        logger.info(f"Generating Morgan fingerprints (radius={radius}, bits={n_bits})...")
        
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.Chem import rdMolDescriptors
        except ImportError:
            logger.warning("RDKit not available. Using simplified fingerprints.")
            return self._generate_simplified_fingerprints(smiles_series, n_bits)
        
        fingerprints_df = pd.DataFrame(index=smiles_series.index)
        
        for idx, smiles in smiles_series.items():
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    # Generate Morgan fingerprint
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    # Convert to array
                    fp_array = np.zeros(n_bits)
                    for bit in fp.GetOnBits():
                        fp_array[bit] = 1
                    fingerprints_df.loc[idx] = fp_array
                else:
                    # Invalid SMILES - use zero vector
                    fingerprints_df.loc[idx] = np.zeros(n_bits)
            except Exception as e:
                logger.warning(f"Error processing SMILES {smiles}: {e}")
                fingerprints_df.loc[idx] = np.zeros(n_bits)
        
        # Name the columns
        fingerprints_df.columns = [f'morgan_{i:04d}' for i in range(n_bits)]
        
        logger.info(f"Generated Morgan fingerprints: {fingerprints_df.shape}")
        return fingerprints_df
    
    def _generate_simplified_fingerprints(self, smiles_series: pd.Series, 
                                        n_bits: int = 256) -> pd.DataFrame:
        """
        Generate simplified fingerprints when RDKit is not available
        
        Args:
            smiles_series: Series containing SMILES strings
            n_bits: Number of bits in fingerprint
            
        Returns:
            DataFrame with simplified fingerprints
        """
        import re
        logger.info("Generating simplified fingerprints (RDKit not available)...")
        
        fingerprints_df = pd.DataFrame(index=smiles_series.index)
        
        # Create simplified features based on SMILES patterns
        for i in range(n_bits):
            col_name = f'simplified_{i:03d}'
            
            # Use different patterns for different bits
            if i < 64:
                # Element-based patterns
                element = ['C', 'N', 'O', 'H', 'F', 'Cl', 'Br', 'I'][i % 8]
                fingerprints_df[col_name] = smiles_series.str.count(element) % 2
            elif i < 128:
                # Bond-based patterns
                bond = ['-', '=', '#', '(', ')', '[', ']', '*'][i % 8]
                # Escape special regex characters
                escaped_bond = re.escape(bond)
                fingerprints_df[col_name] = smiles_series.str.count(escaped_bond) % 2
            elif i < 192:
                # Length-based patterns
                fingerprints_df[col_name] = (smiles_series.str.len() + i) % 2
            else:
                # Random-like patterns
                np.random.seed(i)
                fingerprints_df[col_name] = np.random.choice([0, 1], size=len(smiles_series))
        
        logger.info(f"Generated simplified fingerprints: {fingerprints_df.shape}")
        return fingerprints_df
    
    def create_custom_polymer_features(self, smiles_series: pd.Series, 
                                     descriptors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create custom polymer-specific features
        
        Args:
            smiles_series: Series containing SMILES strings
            descriptors_df: DataFrame with molecular descriptors
            
        Returns:
            DataFrame with custom polymer features
        """
        logger.info("Creating custom polymer-specific features...")
        
        custom_features = pd.DataFrame(index=smiles_series.index)
        
        # Polymer architecture features
        custom_features['is_linear'] = (descriptors_df['connection_points'] == 2).astype(int)
        custom_features['is_branched'] = (descriptors_df['connection_points'] > 2).astype(int)
        custom_features['is_cyclic'] = (descriptors_df['ring_count'] > 0).astype(int)
        
        # Chemical composition ratios
        custom_features['c_n_ratio'] = (descriptors_df['carbon_count'] / 
                                      (descriptors_df['nitrogen_count'] + 1))
        custom_features['c_o_ratio'] = (descriptors_df['carbon_count'] / 
                                      (descriptors_df['oxygen_count'] + 1))
        custom_features['h_c_ratio'] = (descriptors_df['hydrogen_count'] / 
                                      (descriptors_df['carbon_count'] + 1))
        
        # Polarity indicators
        custom_features['polar_atoms'] = (descriptors_df['nitrogen_count'] + 
                                        descriptors_df['oxygen_count'] + 
                                        descriptors_df['fluorine_count'] + 
                                        descriptors_df['chlorine_count'])
        custom_features['polarity_score'] = (custom_features['polar_atoms'] / 
                                           (descriptors_df['carbon_count'] + 1))
        
        # Flexibility indicators
        custom_features['flexibility_score'] = (descriptors_df['single_bonds'] / 
                                              (descriptors_df['double_bonds'] + 
                                               descriptors_df['triple_bonds'] + 1))
        
        # Aromaticity indicators (rough estimate)
        custom_features['aromatic_potential'] = (descriptors_df['ring_count'] * 
                                               descriptors_df['carbon_count'] / 
                                               (descriptors_df['smiles_length'] + 1))
        
        # Fill NaN values
        custom_features = custom_features.fillna(0)
        
        logger.info(f"Created {len(custom_features.columns)} custom polymer features")
        return custom_features
    
    def implement_feature_selection(self, feature_matrix: pd.DataFrame, 
                                  target_data: pd.DataFrame,
                                  method: str = 'correlation',
                                  threshold: float = 0.01,
                                  max_features: int = 1000) -> Tuple[pd.DataFrame, List[str]]:
        """
        Implement feature selection
        
        Args:
            feature_matrix: Feature matrix
            target_data: Target variables
            method: Selection method ('correlation', 'mutual_info', 'variance')
            threshold: Selection threshold
            max_features: Maximum number of features to select
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        logger.info(f"Implementing feature selection using {method} method...")
        
        if method == 'correlation':
            selected_features = self._correlation_based_selection(
                feature_matrix, target_data, threshold, max_features
            )
        elif method == 'mutual_info':
            selected_features = self._mutual_info_selection(
                feature_matrix, target_data, max_features
            )
        elif method == 'variance':
            selected_features = self._variance_based_selection(
                feature_matrix, threshold, max_features
            )
        else:
            logger.warning(f"Unknown method {method}. Using correlation-based selection.")
            selected_features = self._correlation_based_selection(
                feature_matrix, target_data, threshold, max_features
            )
        
        self.selected_features = selected_features
        logger.info(f"Selected {len(selected_features)} features out of {len(feature_matrix.columns)}")
        
        return feature_matrix[selected_features], selected_features
    
    def _correlation_based_selection(self, feature_matrix: pd.DataFrame, 
                                   target_data: pd.DataFrame,
                                   threshold: float,
                                   max_features: int) -> List[str]:
        """Correlation-based feature selection"""
        correlations = {}
        
        for target in self.target_columns:
            if target in target_data.columns:
                target_correlations = feature_matrix.corrwith(target_data[target]).abs()
                correlations[target] = target_correlations
        
        # Combine correlations across all targets
        if correlations:
            combined_correlations = pd.concat(correlations.values(), axis=1).max(axis=1)
            selected = combined_correlations[combined_correlations > threshold].index.tolist()
            
            # Limit to max_features
            if len(selected) > max_features:
                selected = combined_correlations.nlargest(max_features).index.tolist()
        else:
            selected = []
        
        return selected
    
    def _mutual_info_selection(self, feature_matrix: pd.DataFrame, 
                             target_data: pd.DataFrame,
                             max_features: int) -> List[str]:
        """Mutual information-based feature selection"""
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Use the target with most data for mutual info
            best_target = None
            best_coverage = 0
            
            for target in self.target_columns:
                if target in target_data.columns:
                    coverage = target_data[target].notna().sum()
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_target = target
            
            if best_target is None:
                return []
            
            # Calculate mutual information
            target_values = target_data[best_target].dropna()
            feature_subset = feature_matrix.loc[target_values.index]
            
            mi_scores = mutual_info_regression(feature_subset, target_values, random_state=self.random_state)
            mi_series = pd.Series(mi_scores, index=feature_subset.columns)
            
            selected = mi_series.nlargest(max_features).index.tolist()
            return selected
            
        except ImportError:
            logger.warning("sklearn not available for mutual info selection. Using correlation.")
            return self._correlation_based_selection(feature_matrix, target_data, 0.01, max_features)
    
    def _variance_based_selection(self, feature_matrix: pd.DataFrame, 
                                 threshold: float,
                                 max_features: int) -> List[str]:
        """Variance-based feature selection"""
        variances = feature_matrix.var()
        selected = variances[variances > threshold].index.tolist()
        
        if len(selected) > max_features:
            selected = variances.nlargest(max_features).index.tolist()
        
        return selected
    
    def create_final_feature_matrix(self, data: pd.DataFrame, 
                                  smiles_column: str = 'SMILES',
                                  include_targets: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Create final feature matrix for machine learning
        
        Args:
            data: Input DataFrame
            smiles_column: Name of SMILES column
            include_targets: Whether to include target variables
            
        Returns:
            Tuple of (feature_matrix, feature_info)
        """
        logger.info("Creating final feature matrix...")
        
        if smiles_column not in data.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in data")
        
        # Extract molecular descriptors
        molecular_descriptors = self.extract_molecular_descriptors(data[smiles_column])
        
        # Generate Morgan fingerprints
        morgan_fingerprints = self.generate_morgan_fingerprints(data[smiles_column])
        
        # Create custom polymer features
        custom_features = self.create_custom_polymer_features(data[smiles_column], molecular_descriptors)
        
        # Combine all features
        feature_matrix = pd.concat([
            molecular_descriptors,
            morgan_fingerprints,
            custom_features
        ], axis=1)
        
        # Add target variables if requested
        if include_targets:
            for target in self.target_columns:
                if target in data.columns:
                    feature_matrix[target] = data[target]
        
        # Store feature information
        feature_info = {
            'total_features': len(feature_matrix.columns),
            'molecular_descriptors': len(molecular_descriptors.columns),
            'morgan_fingerprints': len(morgan_fingerprints.columns),
            'custom_features': len(custom_features.columns),
            'targets_included': include_targets,
            'feature_names': list(feature_matrix.columns)
        }
        
        self.feature_stats = feature_info
        self.feature_names = list(feature_matrix.columns)
        
        logger.info(f"Final feature matrix created: {feature_matrix.shape}")
        return feature_matrix, feature_info
    
    def get_feature_engineering_summary(self) -> Dict:
        """Get comprehensive feature engineering summary"""
        summary = {
            'feature_engineering_stats': self.feature_stats,
            'selected_features': self.selected_features,
            'total_features_created': len(self.feature_names),
            'random_state': self.random_state
        }
        return summary
    
    def save_feature_matrix(self, feature_matrix: pd.DataFrame, 
                          output_dir: str = "data/processed",
                          filename: str = "feature_matrix.csv") -> str:
        """
        Save feature matrix to file
        
        Args:
            feature_matrix: Feature matrix DataFrame
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        feature_matrix.to_csv(file_path, index=False)
        
        logger.info(f"Feature matrix saved to: {file_path}")
        return str(file_path)

def engineer_polymer_features(data: pd.DataFrame, 
                           smiles_column: str = 'SMILES',
                           feature_selection_method: str = 'correlation',
                           feature_selection_threshold: float = 0.01,
                           max_features: int = 1000,
                           random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function for complete feature engineering
    
    Args:
        data: Input DataFrame
        smiles_column: Name of SMILES column
        feature_selection_method: Method for feature selection
        feature_selection_threshold: Threshold for feature selection
        max_features: Maximum number of features to select
        random_state: Random seed
        
    Returns:
        Tuple of (feature_matrix, feature_engineering_info)
    """
    engineer = PolymerFeatureEngineer(random_state=random_state)
    
    # Create feature matrix
    feature_matrix, feature_info = engineer.create_final_feature_matrix(
        data, smiles_column, include_targets=True
    )
    
    # Implement feature selection
    selected_features, selected_names = engineer.implement_feature_selection(
        feature_matrix, 
        data[engineer.target_columns] if all(col in data.columns for col in engineer.target_columns) else pd.DataFrame(),
        method=feature_selection_method,
        threshold=feature_selection_threshold,
        max_features=max_features
    )
    
    # Get complete summary
    feature_engineering_summary = engineer.get_feature_engineering_summary()
    
    return selected_features, feature_engineering_summary
