# Cluster 1: Data Understanding & Exploration

## Overview

This cluster implements the first phase of the CRISP-DM methodology for the NeurIPS Open Polymer Prediction 2025 competition. It focuses on understanding the data structure, quality, and characteristics before proceeding to modeling.

## 🎯 Target Properties

The competition requires predicting five key polymer properties:
- **Tg** - Glass transition temperature
- **FFV** - Fractional free volume  
- **Tc** - Thermal conductivity
- **Density** - Polymer density
- **Rg** - Radius of gyration

## 📁 Project Structure

```
├── src/
│   ├── data/
│   │   ├── __init__.py          # Data module package
│   │   └── loader.py            # Data loading and validation
│   └── utils/
│       ├── __init__.py          # Utils module package
│       └── visualization.py     # Data visualization utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Jupyter notebook (if available)
│   └── 01_data_exploration.py     # Python script version
├── test_exploration.py          # Test script for modules
└── CLUSTER1_README.md           # This file
```

## 🚀 Getting Started

### Prerequisites

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Competition Data**
   - Extract `neurips-open-polymer-prediction-2025.zip` to the `data/` directory
   - Expected files: `train.csv`, `test.csv`, `sample_submission.csv`

### Quick Start

1. **Test the Setup**
   ```bash
   python test_exploration.py
   ```

2. **Run Data Exploration**
   ```bash
   python notebooks/01_data_exploration.py
   ```

## 🔍 What Cluster 1 Does

### Task 1.1: Load and Examine Competition Data Structure
- Automatically detects and loads available data files
- Provides comprehensive dataset information
- Memory usage analysis
- Column structure examination

### Task 1.2: Analyze Target Variables (Tg, FFV, Tc, Density, Rg)
- Statistical summaries for each target
- Missing value analysis
- Distribution characteristics
- Data quality assessment

### Task 1.3: Explore SMILES Molecular Representations
- SMILES string validation
- Format consistency checks
- Sample SMILES examination
- Validation statistics

### Task 1.4: Generate Basic Statistics and Visualizations
- Dataset overview plots
- Memory usage analysis
- Data type summaries
- Missing value patterns

### Task 1.5: Identify Data Quality Issues and Missing Values
- Comprehensive quality assessment
- Duplicate detection
- Outlier identification
- Data integrity checks

### Task 1.6: Generate Comprehensive Exploration Report
- Automated report generation
- Key insights summary
- Quality issue identification
- Next steps recommendations

## 📊 Output and Visualizations

### Automatic Plots Generated
1. **Dataset Overview** - Size, memory, and structure comparison
2. **Target Distributions** - Histograms and statistics for all 5 properties
3. **Correlation Matrix** - Relationships between target variables
4. **Missing Values Heatmap** - Pattern analysis of missing data
5. **Feature Distributions** - Numerical feature analysis
6. **SMILES Validation** - Molecular representation quality
7. **Data Quality Report** - Issue summary and recommendations

### Console Output
- Comprehensive dataset statistics
- Target variable analysis
- SMILES validation results
- Data quality assessment
- Key insights and recommendations

## 🛠️ Key Components

### PolymerDataLoader Class
```python
from data.loader import PolymerDataLoader

loader = PolymerDataLoader()
data_files = loader.load_all_data()
target_stats = loader.get_target_statistics()
quality_issues = loader.check_data_quality()
```

**Key Methods:**
- `load_all_data()` - Loads all available datasets
- `get_data_info()` - Comprehensive dataset information
- `validate_smiles()` - SMILES validation and analysis
- `get_target_statistics()` - Target variable statistics
- `check_data_quality()` - Data quality assessment

### DataVisualizer Class
```python
from utils.visualization import DataVisualizer

visualizer = DataVisualizer()
visualizer.plot_data_overview(data_info)
visualizer.plot_target_distributions(target_stats)
```

**Key Methods:**
- `plot_data_overview()` - Dataset comparison plots
- `plot_target_distributions()` - Target variable analysis
- `plot_target_correlations()` - Correlation analysis
- `plot_missing_values_heatmap()` - Missing data patterns
- `plot_feature_distributions()` - Feature analysis
- `plot_smiles_analysis()` - SMILES validation plots
- `plot_data_quality_report()` - Quality issue summary

## 📈 Expected Output

When you run the exploration script, you'll see:

```
================================================================================
NEURIPS OPEN POLYMER PREDICTION 2025 - DATA EXPLORATION
CLUSTER 1: Data Understanding & Exploration
================================================================================

🔍 Initializing data exploration tools...
📁 Checking available data files...
Files in data directory: ['train.csv', 'test.csv', 'sample_submission.csv']

📊 Loading competition data...
Loaded datasets: ['train', 'test', 'sample_submission']

🎯 Analyzing target variables...
Found 5 target variables:
  Tg: Count: 10,000, Range: [150.0, 450.0], Mean: 300.0, Std: 50.0...

🧪 Validating SMILES molecular representations...
SMILES Validation Results:
  total_smiles: 10,000
  valid_smiles: 9,950
  invalid_smiles: 50
  empty_smiles: 0

📊 Generating dataset statistics...
Dataset Overview:
  TRAIN Dataset:
    Shape: (10000, 7)
    Memory: 2.45 MB
    Columns: ['id', 'SMILES', 'Tg', 'FFV', 'Tc', 'Density', 'Rg']
...
```

## 🔧 Customization

### Adding Custom Analysis
You can extend the exploration by adding custom methods to the classes:

```python
# Add custom analysis to PolymerDataLoader
def custom_analysis(self):
    # Your custom analysis code
    pass

# Add custom visualizations to DataVisualizer
def custom_plot(self, data):
    # Your custom plotting code
    pass
```

### Modifying Target Properties
If you need to analyze different properties, modify the target columns in `loader.py`:

```python
target_columns = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']  # Modify as needed
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the project root directory
   cd "NeurIPS - Open Polymer Prediction 2025"
   python test_exploration.py
   ```

2. **Data Not Found**
   - Ensure competition data is extracted to `data/` directory
   - Check file names: `train.csv`, `test.csv`, `sample_submission.csv`

3. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Visualization Issues**
   - For Jupyter notebooks: `%matplotlib inline`
   - For headless environments: `plt.switch_backend('Agg')`

### Getting Help

1. **Run the test script first**: `python test_exploration.py`
2. **Check the console output** for specific error messages
3. **Verify data files** are in the correct location
4. **Ensure all dependencies** are installed

## 📚 Next Steps

After completing Cluster 1, you'll be ready for:

**Cluster 2: Data Preparation & Cleaning**
- Handle missing values
- Clean SMILES strings
- Remove duplicates and outliers
- Data standardization

**Cluster 3: Feature Engineering**
- Extract molecular descriptors
- Generate Morgan fingerprints
- Create custom features
- Feature selection

**Cluster 4: Traditional Machine Learning**
- Baseline models
- Ensemble methods
- Hyperparameter tuning
- Cross-validation

## 🎉 Success Criteria

Cluster 1 is successful when you have:
- ✅ All data files loaded and validated
- ✅ Target variable statistics computed
- ✅ SMILES validation completed
- ✅ Data quality issues identified
- ✅ Comprehensive visualizations generated
- ✅ Exploration report completed
- ✅ Clear understanding of data structure and quality

## 📊 Performance Notes

- **Memory Usage**: The loader is optimized for large datasets
- **Speed**: Basic analysis completes in seconds for typical competition datasets
- **Scalability**: Handles datasets up to several GB efficiently
- **Visualization**: Plots are optimized for both screen and file output

---

**Ready to explore your polymer data? Run `python notebooks/01_data_exploration.py` to get started!** 🚀
