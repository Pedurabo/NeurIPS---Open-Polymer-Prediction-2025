"""
Visualization utilities for polymer prediction data exploration
Provides comprehensive plotting functions for understanding data structure and relationships
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Configure plotting style
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class DataVisualizer:
    """Comprehensive data visualization class for polymer prediction data"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
        
    def plot_data_overview(self, data_info: Dict[str, Dict]) -> None:
        """
        Create an overview plot showing dataset information
        
        Args:
            data_info: Dictionary containing dataset information from loader
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # Dataset sizes
        datasets = list(data_info.keys())
        sizes = [data_info[d]['shape'][0] for d in datasets]
        axes[0, 0].bar(datasets, sizes, color=self.colors[:len(datasets)])
        axes[0, 0].set_title('Dataset Sizes (Rows)')
        axes[0, 0].set_ylabel('Number of Rows')
        
        # Column counts
        col_counts = [data_info[d]['shape'][1] for d in datasets]
        axes[0, 1].bar(datasets, col_counts, color=self.colors[:len(datasets)])
        axes[0, 1].set_title('Dataset Sizes (Columns)')
        axes[0, 1].set_ylabel('Number of Columns')
        
        # Memory usage
        memory_usage = [data_info[d]['memory_usage'] / 1024**2 for d in datasets]  # Convert to MB
        axes[1, 0].bar(datasets, memory_usage, color=self.colors[:len(datasets)])
        axes[1, 0].set_title('Memory Usage')
        axes[1, 0].set_ylabel('Memory (MB)')
        
        # Null value counts
        total_nulls = [sum(data_info[d]['null_counts'].values()) for d in datasets]
        axes[1, 1].bar(datasets, total_nulls, color=self.colors[:len(datasets)])
        axes[1, 1].set_title('Total Missing Values')
        axes[1, 1].set_ylabel('Number of Missing Values')
        
        plt.tight_layout()
        plt.show()
        
    def plot_target_distributions(self, target_stats: Dict[str, Dict]) -> None:
        """
        Plot distributions of target variables
        
        Args:
            target_stats: Dictionary containing target variable statistics
        """
        if not target_stats:
            print("No target statistics available")
            return
            
        n_targets = len(target_stats)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Target Variable Distributions', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes_flat = axes.flatten()
        
        for i, (target, stats) in enumerate(target_stats.items()):
            if i >= len(axes_flat):
                break
                
            ax = axes_flat[i]
            
            # Create synthetic data for visualization based on statistics
            if stats['count'] > 0:
                # Generate sample data that matches the statistics
                mean, std = stats['mean'], stats['std']
                min_val, max_val = stats['min'], stats['max']
                
                # Create a reasonable distribution
                if std > 0:
                    # Normal distribution if std > 0
                    sample_data = np.random.normal(mean, std, 1000)
                    sample_data = np.clip(sample_data, min_val, max_val)
                else:
                    # Uniform distribution if std = 0
                    sample_data = np.random.uniform(min_val, max_val, 1000)
                    
                ax.hist(sample_data, bins=30, alpha=0.7, color=self.colors[i % len(self.colors)])
                ax.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
                ax.axvline(stats['median'], color='green', linestyle='--', label=f'Median: {stats["median"]:.3f}')
                
            ax.set_title(f'{target} Distribution')
            ax.set_xlabel(target)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Hide unused subplots
        for i in range(n_targets, len(axes_flat)):
            axes_flat[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def plot_target_correlations(self, train_data: pd.DataFrame) -> None:
        """
        Plot correlation matrix for target variables
        
        Args:
            train_data: Training dataset
        """
        target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
        available_targets = [col for col in target_columns if col in train_data.columns]
        
        if len(available_targets) < 2:
            print("Need at least 2 target variables for correlation analysis")
            return
            
        # Calculate correlations
        target_data = train_data[available_targets].dropna()
        corr_matrix = target_data.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.3f')
        plt.title('Target Variable Correlations', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print correlation insights
        print("\nCorrelation Insights:")
        for i, target1 in enumerate(available_targets):
            for j, target2 in enumerate(available_targets[i+1:], i+1):
                corr_val = corr_matrix.iloc[i, j]
                strength = "strong" if abs(corr_val) > 0.7 else "moderate" if abs(corr_val) > 0.3 else "weak"
                direction = "positive" if corr_val > 0 else "negative"
                print(f"{target1} vs {target2}: {corr_val:.3f} ({strength} {direction} correlation)")
                
    def plot_smiles_analysis(self, smiles_validation: Dict[str, int]) -> None:
        """
        Plot SMILES validation results
        
        Args:
            smiles_validation: Dictionary containing SMILES validation results
        """
        if not smiles_validation:
            print("No SMILES validation data available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('SMILES Validation Analysis', fontsize=16, fontweight='bold')
        
        # Pie chart of validation results
        labels = ['Valid', 'Invalid', 'Empty']
        sizes = [smiles_validation['valid_smiles'], 
                smiles_validation['invalid_smiles'], 
                smiles_validation['empty_smiles']]
        colors = ['lightgreen', 'lightcoral', 'lightblue']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('SMILES Validation Results')
        
        # Bar chart of counts
        ax2.bar(labels, sizes, color=colors)
        ax2.set_title('SMILES Counts by Validation Status')
        ax2.set_ylabel('Count')
        
        # Add value labels on bars
        for i, v in enumerate(sizes):
            ax2.text(i, v + max(sizes) * 0.01, str(v), ha='center', va='bottom')
            
        plt.tight_layout()
        plt.show()
        
    def plot_missing_values_heatmap(self, train_data: pd.DataFrame) -> None:
        """
        Create a heatmap showing missing values pattern
        
        Args:
            train_data: Training dataset
        """
        plt.figure(figsize=(12, 8))
        
        # Create missing values matrix
        missing_matrix = train_data.isnull()
        
        # Plot heatmap
        sns.heatmap(missing_matrix, 
                   cbar=True, 
                   yticklabels=False,
                   cmap='viridis')
        plt.title('Missing Values Pattern', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Print missing values summary
        missing_summary = train_data.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        
        if not missing_summary.empty:
            print("\nMissing Values Summary:")
            for col, count in missing_summary.items():
                percentage = (count / len(train_data)) * 100
                print(f"{col}: {count} ({percentage:.2f}%)")
        else:
            print("\nNo missing values found in the dataset!")
            
    def plot_feature_distributions(self, train_data: pd.DataFrame, n_features: int = 10) -> None:
        """
        Plot distributions of numerical features
        
        Args:
            train_data: Training dataset
            n_features: Number of features to plot
        """
        # Select numerical columns (exclude SMILES and ID columns)
        numerical_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            print("No numerical features found")
            return
            
        # Limit to n_features for readability
        if len(numerical_cols) > n_features:
            numerical_cols = numerical_cols[:n_features]
            
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
            
        for i, col in enumerate(numerical_cols):
            if i < len(axes):
                ax = axes[i]
                
                # Plot histogram
                ax.hist(train_data[col].dropna(), bins=30, alpha=0.7, color=self.colors[i % len(self.colors)])
                ax.set_title(f'{col} Distribution')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
        # Hide unused subplots
        for i in range(len(numerical_cols), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
        
    def create_interactive_dashboard(self, train_data: pd.DataFrame, target_stats: Dict[str, Dict]) -> None:
        """
        Create an interactive Plotly dashboard for data exploration
        
        Args:
            train_data: Training dataset
            target_stats: Dictionary containing target variable statistics
        """
        if not target_stats:
            print("No target statistics available for interactive dashboard")
            return
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(target_stats.keys()),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "violin"}]]
        )
        
        # Add histogram for first target
        first_target = list(target_stats.keys())[0]
        if first_target in train_data.columns:
            fig.add_trace(
                go.Histogram(x=train_data[first_target].dropna(), name=f"{first_target} Histogram"),
                row=1, col=1
            )
            
        # Add box plot for second target
        if len(target_stats) > 1:
            second_target = list(target_stats.keys())[1]
            if second_target in train_data.columns:
                fig.add_trace(
                    go.Box(y=train_data[second_target].dropna(), name=f"{second_target} Box Plot"),
                    row=1, col=2
                )
                
        # Add scatter plot for third target
        if len(target_stats) > 2:
            third_target = list(target_stats.keys())[2]
            if third_target in train_data.columns:
                fig.add_trace(
                    go.Scatter(x=train_data.index, y=train_data[third_target].dropna(), 
                              mode='markers', name=f"{third_target} Scatter"),
                    row=2, col=1
                )
                
        # Add violin plot for fourth target
        if len(target_stats) > 3:
            fourth_target = list(target_stats.keys())[3]
            if fourth_target in train_data.columns:
                fig.add_trace(
                    go.Violin(y=train_data[fourth_target].dropna(), name=f"{fourth_target} Violin"),
                    row=2, col=2
                )
                
        fig.update_layout(height=800, title_text="Interactive Data Exploration Dashboard")
        fig.show()
        
    def plot_data_quality_report(self, quality_issues: Dict[str, List[str]]) -> None:
        """
        Create a visual report of data quality issues
        
        Args:
            quality_issues: Dictionary containing quality issues
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Data Quality Report', fontsize=16, fontweight='bold')
        
        # Count issues by type
        issue_types = list(quality_issues.keys())
        issue_counts = [len(quality_issues[issue_type]) for issue_type in issue_types]
        colors = ['red', 'orange', 'blue']
        
        # Bar chart of issue counts
        axes[0].bar(issue_types, issue_counts, color=colors)
        axes[0].set_title('Data Quality Issues by Type')
        axes[0].set_ylabel('Number of Issues')
        
        # Add value labels on bars
        for i, v in enumerate(issue_counts):
            axes[0].text(i, v + 0.1, str(v), ha='center', va='bottom')
            
        # Pie chart of issue distribution
        if sum(issue_counts) > 0:
            axes[1].pie(issue_counts, labels=issue_types, colors=colors, autopct='%1.1f%%')
            axes[1].set_title('Issue Distribution')
        else:
            axes[1].text(0.5, 0.5, 'No Issues Found!', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Issue Distribution')
            
        # Text summary
        axes[2].axis('off')
        summary_text = "Data Quality Summary:\n\n"
        
        for issue_type, issues in quality_issues.items():
            summary_text += f"{issue_type.title()}:\n"
            if issues:
                for issue in issues[:5]:  # Show first 5 issues
                    summary_text += f"• {issue}\n"
                if len(issues) > 5:
                    summary_text += f"• ... and {len(issues) - 5} more\n"
            else:
                summary_text += "• None found\n"
            summary_text += "\n"
            
        axes[2].text(0.1, 0.9, summary_text, transform=axes[2].transAxes, 
                     fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()

def create_exploration_report(loader, visualizer) -> None:
    """
    Create a comprehensive exploration report
    
    Args:
        loader: PolymerDataLoader instance
        visualizer: DataVisualizer instance
    """
    print("=" * 60)
    print("COMPREHENSIVE DATA EXPLORATION REPORT")
    print("=" * 60)
    
    # Load data
    print("\n1. LOADING DATA...")
    data_files = loader.load_all_data()
    
    if not data_files:
        print("No data files found. Please ensure data is in the data/ directory.")
        return
        
    # Get data information
    print("\n2. DATA INFORMATION...")
    data_info = loader.get_data_info()
    for dataset_name, info in data_info.items():
        print(f"\n{dataset_name.upper()} Dataset:")
        print(f"  Shape: {info['shape']}")
        print(f"  Columns: {info['columns']}")
        print(f"  Memory: {info['memory_usage'] / 1024**2:.2f} MB")
        
    # Validate SMILES
    print("\n3. SMILES VALIDATION...")
    smiles_validation = loader.validate_smiles()
    for key, value in smiles_validation.items():
        print(f"  {key}: {value}")
        
    # Get target statistics
    print("\n4. TARGET VARIABLE STATISTICS...")
    target_stats = loader.get_target_statistics()
    for target, stats in target_stats.items():
        print(f"\n  {target}:")
        print(f"    Count: {stats['count']}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Std: {stats['std']:.4f}")
        print(f"    Missing: {stats['null_count']}")
        
    # Check data quality
    print("\n5. DATA QUALITY ASSESSMENT...")
    quality_issues = loader.check_data_quality()
    for issue_type, issues in quality_issues.items():
        if issues:
            print(f"\n  {issue_type.title()}:")
            for issue in issues:
                print(f"    • {issue}")
                
    # Create visualizations
    print("\n6. CREATING VISUALIZATIONS...")
    if data_info:
        visualizer.plot_data_overview(data_info)
        
    if target_stats:
        visualizer.plot_target_distributions(target_stats)
        
    if 'train' in data_files and data_files['train'] is not None:
        visualizer.plot_target_correlations(data_files['train'])
        visualizer.plot_missing_values_heatmap(data_files['train'])
        visualizer.plot_feature_distributions(data_files['train'])
        
    if smiles_validation:
        visualizer.plot_smiles_analysis(smiles_validation)
        
    if quality_issues:
        visualizer.plot_data_quality_report(quality_issues)
        
    print("\n" + "=" * 60)
    print("EXPLORATION REPORT COMPLETE!")
    print("=" * 60)
