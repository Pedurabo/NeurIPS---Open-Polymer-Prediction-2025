# NeurIPS - Open Polymer Prediction 2025

Predicting polymer properties with machine learning to accelerate sustainable materials research.

## Competition Overview

This competition challenges participants to predict fundamental polymer properties directly from chemical structure (SMILES notation). The goal is to accelerate sustainable polymer research through virtual screening and drive advancements in materials science.

## Target Properties

The model must predict five key polymer properties:
- **Tg** - Glass transition temperature
- **FFV** - Fractional free volume
- **Tc** - Thermal conductivity  
- **Density** - Polymer density
- **Rg** - Radius of gyration

## Evaluation Metric

Weighted Mean Absolute Error (wMAE) across all five properties, with reweighting factors to ensure equal contribution regardless of scale or frequency.

## Project Structure

```
├── data/                   # Data files (train, test, sample submission)
├── notebooks/             # Jupyter notebooks for exploration and modeling
├── src/                   # Source code modules
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # ML model implementations
│   └── utils/            # Utility functions
├── models/                # Saved model files
├── submissions/           # Competition submissions
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Download competition data to `data/` directory
3. Run exploration notebooks in `notebooks/`
4. Train models using scripts in `src/`
5. Generate predictions and submit to competition

## Timeline

- **Start Date**: June 16, 2025
- **Entry Deadline**: September 8, 2025
- **Team Merger Deadline**: September 8, 2025
- **Final Submission**: September 15, 2025

## Prizes

- 1st Place: $12,000
- 2nd Place: $10,000
- 3rd Place: $10,000
- 4th Place: $8,000
- 5th Place: $5,000
- Top Student Group: $5,000

## Code Requirements

- CPU/GPU Notebook ≤ 9 hours run-time
- Internet access disabled
- Freely & publicly available external data allowed
- Submission file must be named `submission.csv`

## Citation

Gang Liu, Jiaxin Xu, Eric Inae, Yihan Zhu, Ying Li, Tengfei Luo, Meng Jiang, Yao Yan, Walter Reade, Sohier Dane, Addison Howard, and María Cruz. NeurIPS - Open Polymer Prediction 2025. https://kaggle.com/competitions/neurips-open-polymer-prediction-2025, 2025. Kaggle.
