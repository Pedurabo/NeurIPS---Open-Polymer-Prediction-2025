"""
Supplementary Data Integration for NeurIPS Open Polymer Prediction 2025
Intelligently combines main training data with supplementary datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SupplementDataIntegrator:
    """Intelligently integrates supplementary datasets with main training data"""
    
    def __init__(self):
        """Initialize the integrator"""
        self.main_columns = ['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
        self.target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        
    def integrate_supplementary_data(self, main_data: pd.DataFrame, 
                                   supplement_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Integrate supplementary datasets with main training data
        
        Args:
            main_data: Main training DataFrame
            supplement_data: Dictionary of supplementary datasets
            
        Returns:
            Integrated DataFrame with maximum data utilization
        """
        logger.info("Starting supplementary data integration...")
        
        # Start with main data
        integrated_data = main_data.copy()
        integration_stats = {
            'main_samples': len(main_data),
            'supplement_samples_added': 0,
            'target_coverage_improvements': {},
            'total_integrated_samples': len(main_data)
        }
        
        # Track which targets we're improving
        target_improvements = {target: 0 for target in self.target_columns}
        
        # Process each supplementary dataset
        for dataset_name, supp_data in supplement_data.items():
            logger.info(f"Processing {dataset_name}: {supp_data.shape}")
            
            # Strategy 1: Direct target property addition
            added_samples = self._add_target_properties(integrated_data, supp_data, dataset_name)
            if added_samples > 0:
                integration_stats['supplement_samples_added'] += added_samples
                integration_stats['total_integrated_samples'] += added_samples
                
                # Track target improvements
                for target in self.target_columns:
                    if target in supp_data.columns:
                        target_improvements[target] += added_samples
        
        # Strategy 2: Feature engineering from SMILES-only datasets
        smiles_only_datasets = {k: v for k, v in supplement_data.items() 
                              if len(v.columns) == 1 and 'SMILES' in v.columns}
        
        for dataset_name, supp_data in smiles_only_datasets.items():
            logger.info(f"Processing SMILES-only dataset: {dataset_name}")
            # These will be used for feature engineering in Cluster 3
            # For now, we'll note their availability
            integration_stats[f'{dataset_name}_smiles_samples'] = len(supp_data)
        
        # Calculate final target coverage improvements
        for target in self.target_columns:
            if target in integrated_data.columns:
                original_coverage = main_data[target].notna().sum()
                final_coverage = integrated_data[target].notna().sum()
                improvement = final_coverage - original_coverage
                
                if improvement > 0:
                    integration_stats['target_coverage_improvements'][target] = {
                        'original': original_coverage,
                        'final': final_coverage,
                        'improvement': improvement,
                        'improvement_pct': (improvement / original_coverage * 100) if original_coverage > 0 else 0
                    }
        
        logger.info(f"Integration complete. Total samples: {len(integrated_data)}")
        return integrated_data, integration_stats
    
    def _add_target_properties(self, integrated_data: pd.DataFrame, 
                              supp_data: pd.DataFrame, 
                              dataset_name: str) -> int:
        """
        Add target properties from supplementary dataset to main data
        
        Args:
            integrated_data: Main integrated DataFrame
            supp_data: Supplementary dataset
            dataset_name: Name of the supplementary dataset
            
        Returns:
            Number of samples added
        """
        added_samples = 0
        
        # Check what target properties this dataset has
        available_targets = [col for col in supp_data.columns if col in self.target_columns]
        
        if not available_targets:
            logger.info(f"  {dataset_name}: No target properties to add")
            return 0
        
        logger.info(f"  {dataset_name}: Adding targets: {available_targets}")
        
        # For each target property, try to add samples
        for target in available_targets:
            # Find samples in supplementary data that have this target
            valid_supp_samples = supp_data[['SMILES', target]].dropna()
            
            if len(valid_supp_samples) == 0:
                continue
            
            # Find samples in main data that are missing this target
            missing_main_samples = integrated_data[integrated_data[target].isna()]
            
            if len(missing_main_samples) == 0:
                logger.info(f"    {target}: No missing values in main data")
                continue
            
            # Try to match SMILES and add target values
            matches_found = 0
            
            # Create a mapping from SMILES to target values for efficiency
            smiles_to_target = dict(zip(valid_supp_samples['SMILES'], valid_supp_samples[target]))
            
            # Find samples in main data that are missing this target
            missing_main_samples = integrated_data[integrated_data[target].isna()]
            
            if len(missing_main_samples) == 0:
                logger.info(f"    {target}: No missing values in main data")
                continue
            
            # Check for exact SMILES matches
            for idx, row in missing_main_samples.iterrows():
                main_smiles = row['SMILES']
                if main_smiles in smiles_to_target:
                    # Update the target value
                    integrated_data.loc[idx, target] = smiles_to_target[main_smiles]
                    matches_found += 1
            
            if matches_found > 0:
                logger.info(f"    {target}: Added {matches_found} values from {dataset_name}")
                added_samples += matches_found
            else:
                logger.info(f"    {target}: No SMILES matches found with {dataset_name}")
        
        return added_samples
    
    def get_integration_summary(self, main_data: pd.DataFrame, 
                               integrated_data: pd.DataFrame,
                               integration_stats: Dict) -> Dict:
        """
        Generate comprehensive integration summary
        
        Args:
            main_data: Original main training data
            integrated_data: Final integrated data
            integration_stats: Integration statistics
            
        Returns:
            Summary dictionary
        """
        summary = {
            'integration_overview': {
                'main_samples': len(main_data),
                'final_samples': len(integrated_data),
                'supplement_samples_utilized': integration_stats['supplement_samples_added'],
                'total_improvement': integration_stats['supplement_samples_added']
            },
            'target_coverage_analysis': {},
            'supplementary_datasets': {}
        }
        
        # Analyze target coverage improvements
        for target in self.target_columns:
            if target in main_data.columns and target in integrated_data.columns:
                original_coverage = main_data[target].notna().sum()
                final_coverage = integrated_data[target].notna().sum()
                improvement = final_coverage - original_coverage
                
                summary['target_coverage_analysis'][target] = {
                    'original_coverage': original_coverage,
                    'original_percentage': (original_coverage / len(main_data)) * 100,
                    'final_coverage': final_coverage,
                    'final_percentage': (final_coverage / len(integrated_data)) * 100,
                    'improvement': improvement,
                    'improvement_percentage': (improvement / len(main_data)) * 100
                }
        
        # Add integration stats
        summary['integration_stats'] = integration_stats
        
        return summary
    
    def save_integrated_data(self, integrated_data: pd.DataFrame, 
                           output_dir: str = "data/processed",
                           filename: str = "integrated_training_data.csv") -> str:
        """
        Save integrated data to file
        
        Args:
            integrated_data: Integrated DataFrame
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        integrated_data.to_csv(file_path, index=False)
        
        logger.info(f"Integrated data saved to: {file_path}")
        return str(file_path)

def integrate_all_supplementary_data(main_data: pd.DataFrame, 
                                   supplement_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to integrate all supplementary data
    
    Args:
        main_data: Main training DataFrame
        supplement_data: Dictionary of supplementary datasets
        
    Returns:
        Tuple of (integrated_data, integration_summary)
    """
    integrator = SupplementDataIntegrator()
    
    # Integrate supplementary data
    integrated_data, integration_stats = integrator.integrate_supplementary_data(
        main_data, supplement_data
    )
    
    # Generate summary
    integration_summary = integrator.get_integration_summary(
        main_data, integrated_data, integration_stats
    )
    
    return integrated_data, integration_summary
