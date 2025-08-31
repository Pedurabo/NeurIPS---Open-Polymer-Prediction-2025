"""
Configuration file for NeurIPS Open Polymer Prediction 2025.

This file contains common settings and parameters used throughout the project.
Modify these values to customize the behavior of the models and preprocessing.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Data files
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

# Target properties
TARGET_PROPERTIES = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# Preprocessing settings
PREPROCESSING_CONFIG = {
    'use_rdkit': True,
    'use_mordred': True,
    'use_fingerprints': True,
    'fingerprint_size': 1024,  # Reduced for faster processing
    'use_3d_descriptors': False,
    'max_smiles_length': 500,  # Truncate very long SMILES
}

# Feature engineering settings
FEATURE_ENGINEERING_CONFIG = {
    'create_interactions': True,
    'create_polynomials': False,  # Disabled for baseline
    'polynomial_degree': 2,
    'use_pca': False,  # Disabled for baseline
    'pca_components': None,  # Auto-determine
    'create_statistical_features': True,
    'create_domain_features': True,
    'max_interactions': 100,  # Limit interaction features
    'rolling_window_sizes': [3, 5, 10],
}

# Model settings
MODEL_CONFIG = {
    'random_state': 42,
    'validation_split': 0.2,
    'cross_validation_folds': 5,
    
    # Random Forest
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'n_jobs': -1,
    },
    
    # Gradient Boosting
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
    },
    
    # XGBoost
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_jobs': -1,
    },
    
    # LightGBM
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_jobs': -1,
    },
}

# Training settings
TRAINING_CONFIG = {
    'early_stopping_rounds': 50,
    'verbose': True,
    'save_models': True,
    'save_predictions': True,
}

# Evaluation settings
EVALUATION_CONFIG = {
    'metrics': ['MAE', 'MSE', 'RMSE', 'R2'],
    'use_weighted_mae': True,  # Competition metric
    'save_plots': True,
}

# Logging settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / 'logs' / 'training.log',
}

# Competition settings
COMPETITION_CONFIG = {
    'name': 'NeurIPS Open Polymer Prediction 2025',
    'evaluation_metric': 'wMAE',  # Weighted Mean Absolute Error
    'submission_format': {
        'required_columns': ['id'] + TARGET_PROPERTIES,
        'filename': 'submission.csv',
    },
    'timeline': {
        'start_date': '2025-06-16',
        'entry_deadline': '2025-09-08',
        'final_submission': '2025-09-15',
    },
}

# Create necessary directories
def create_directories():
    """Create necessary project directories."""
    directories = [DATA_DIR, MODELS_DIR, SUBMISSIONS_DIR, NOTEBOOKS_DIR]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        print(f"✓ Directory created/verified: {directory}")

# Environment variables
def get_env_config():
    """Get configuration from environment variables."""
    return {
        'use_gpu': os.getenv('USE_GPU', 'false').lower() == 'true',
        'num_workers': int(os.getenv('NUM_WORKERS', '4')),
        'memory_limit': os.getenv('MEMORY_LIMIT', '8GB'),
        'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
    }

if __name__ == "__main__":
    print("🔧 Project Configuration")
    print("=" * 40)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Submissions directory: {SUBMISSIONS_DIR}")
    
    print("\n📊 Target Properties:")
    for prop in TARGET_PROPERTIES:
        print(f"  - {prop}")
    
    print("\n⚙️ Feature Engineering:")
    for key, value in FEATURE_ENGINEERING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n🤖 Model Types:")
    for model_type in MODEL_CONFIG.keys():
        if model_type != 'random_state' and model_type != 'validation_split' and model_type != 'cross_validation_folds':
            print(f"  - {model_type}")
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    print("\n✅ Configuration loaded successfully!")
