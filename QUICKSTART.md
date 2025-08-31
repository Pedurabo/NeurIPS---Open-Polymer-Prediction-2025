# 🚀 Quick Start Guide - NeurIPS Open Polymer Prediction 2025

This guide will help you get started with the polymer property prediction project in just a few steps!

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for version control)

## 🛠️ Installation

### 1. Clone or Download the Project
```bash
# Option 1: Clone with Git
git clone <repository-url>
cd "NeurIPS - Open Polymer Prediction 2025"

# Option 2: Download and extract ZIP file
# Then navigate to the project directory
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# If you encounter issues with RDKit, try:
# pip install rdkit-pypi
# pip install mordred
```

### 3. Verify Installation
```bash
# Run the configuration script
python config.py
```

## 📊 Getting Your Data

### 1. Download Competition Data
- Go to the [Kaggle competition page](https://kaggle.com/competitions/neurips-open-polymer-prediction-2025)
- Download the following files:
  - `train.csv` - Training data with SMILES and target properties
  - `test.csv` - Test data with SMILES (no targets)
  - `sample_submission.csv` - Example submission format

### 2. Place Data Files
Put all downloaded CSV files in the `data/` directory:
```
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## 🎯 Quick Start - Run Everything

### Option 1: Complete Pipeline (Recommended)
```bash
# This will run the entire pipeline automatically
python train_baseline.py
```

This script will:
- ✅ Load and preprocess your data
- ✅ Extract molecular features from SMILES
- ✅ Engineer additional features
- ✅ Train multiple baseline models
- ✅ Generate predictions
- ✅ Create a submission file

### Option 2: Step-by-Step Exploration
```bash
# 1. Explore your data
python notebooks/01_data_exploration.py

# 2. Run the training pipeline
python train_baseline.py
```

## 📈 What You'll Get

After running the pipeline, you'll have:

### Models
- `models/preprocessor.joblib` - Fitted data preprocessor
- `models/feature_engineer.joblib` - Fitted feature engineer
- `models/best_baseline_model.joblib` - Best performing model

### Submissions
- `submissions/submission_baseline.csv` - Ready-to-submit predictions

### Logs and Results
- Training performance metrics
- Feature importance analysis
- Cross-validation scores

## 🔍 Understanding the Output

### Model Performance
The script will show you:
- MAE (Mean Absolute Error) for each target property
- R² scores for model fit
- Cross-validation results
- Best model selection

### Target Properties
Your model predicts 5 polymer properties:
- **Tg** - Glass transition temperature
- **FFV** - Fractional free volume
- **Tc** - Thermal conductivity
- **Density** - Polymer density
- **Rg** - Radius of gyration

## 🚀 Next Steps

### 1. Improve Your Model
- Experiment with different feature engineering approaches
- Try advanced models (Neural Networks, Transformers)
- Implement hyperparameter tuning
- Use ensemble methods

### 2. Advanced Features
- Enable 3D molecular descriptors
- Use larger fingerprint sizes
- Implement custom domain features
- Add molecular visualization

### 3. Competition Submission
- Submit your `submission_baseline.csv` to Kaggle
- Monitor your leaderboard position
- Iterate and improve!

## 🆘 Troubleshooting

### Common Issues

#### RDKit Installation Problems
```bash
# Try conda instead of pip
conda install -c conda-forge rdkit

# Or use a pre-built wheel
pip install rdkit-pypi
```

#### Memory Issues
- Reduce `fingerprint_size` in config.py
- Use smaller sample sizes for exploration
- Enable PCA for dimensionality reduction

#### Slow Processing
- Disable Mordred descriptors temporarily
- Use smaller fingerprint sizes
- Process data in batches

### Getting Help
- Check the logs in the console output
- Review the configuration in `config.py`
- Ensure all dependencies are installed
- Verify your data files are in the correct format

## 📚 Learn More

### Project Structure
```
├── data/                   # Your competition data
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering
│   ├── models/            # Machine learning models
│   └── utils/             # Utility functions
├── notebooks/              # Jupyter notebooks
├── models/                 # Saved models
├── submissions/            # Competition submissions
├── config.py              # Configuration settings
├── train_baseline.py      # Main training script
└── requirements.txt       # Python dependencies
```

### Key Files
- `README.md` - Comprehensive project documentation
- `config.py` - All configuration settings
- `train_baseline.py` - Main training pipeline
- `notebooks/01_data_exploration.py` - Data exploration script

## 🎉 Success!

You've successfully set up and run your first polymer property prediction model! 

The baseline models should give you a solid starting point. Now it's time to experiment, improve, and climb the leaderboard!

**Good luck in the competition! 🏆**

---

*Need help? Check the main README.md for detailed documentation, or run `python config.py` to verify your setup.*
