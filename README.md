## Overview
This repository contains an end-to-end, locally run machine learning pipeline designed to understand user emotional states from noisy journal reflections and biological metadata, evaluate uncertainty, and recommend actionable next steps. 

The system strictly adheres to the constraint of running 100% locally, prioritizing user privacy and edge-deployment feasibility without relying on external LLM APIs.

## Approach & Architecture
The system is divided into three core layers:
1. **Data Sanitization & Feature Engineering:** Cleans messy, real-world inputs (e.g., typos in timestamps, missing biological data) across both training and unseen test datasets.
2. **The Machine Learning Core:** Utilizes two distinct XGBoost models trained on 100% of the provided training data to predict the categorical emotional state and the continuous emotional intensity.
3. **The Decision Engine:** A deterministic, rule-based routing layer that takes the model predictions, evaluates confidence, and assigns a specific action (`what_to_do`) and timeline (`when_to_do`).

## Feature Engineering
Real-world data is inherently messy. The pipeline handles this by:
* **Text Processing (TF-IDF):** The unstructured `journal_text` is transformed into a numerical matrix using TF-IDF, capped at the top 500 features to maintain a lightweight footprint. The vectorizer is fitted strictly on the training data to prevent data leakage.
* **Metadata Parsing:** The chaotic `time_of_day` strings are split using regex into continuous numerical features (`time_numeric`) and categorical bins (`time_period`).
* **Robust Imputation:** Missing continuous features (like `sleep_hours`) in the test set are imputed using the training set's median to avoid skewing the distributions. Missing categorical features are explicitly labeled as `unknown`.

## Model Choice
**XGBoost (Extreme Gradient Boosting)** was selected for both the State Classifier and the Intensity Regressor. 
* **Why XGBoost?** It aggressively outperforms deep neural networks on structured/tabular data and natively handles the sparsity introduced by the TF-IDF vectorizer.
* **Why Regression for Intensity?** Intensity (1-5) is treated as a Regression problem rather than Classification because the ordinal distance matters (a prediction of 4 is closer to a true 5 than a 1 is).
* **Uncertainty Awareness:** Instead of building a separate uncertainty model, the system leverages XGBoost's native `predict_proba()`. If the highest probability class falls below a 45% threshold, the system explicitly flags the prediction as uncertain (1).

## Setup & How to Run

### Prerequisites
* Python 3.8+
* Ensure `dataset.csv` (training data) and `test_dataset.csv` (test data) are in the root directory.

### Installation
1. Clone this repository or download the project folder.
2. Install the required local dependencies:
        pip install pandas numpy scikit-learn xgboost 

### Execution
The pipeline is split into two distinct execution steps:

Step 1: Clean the Training Data
Run the preprocessing script to sanitize the raw training inputs:
        python data_prep.py
    (Outputs dataset_cleaned.csv)

Step 2: Train, Process Test Data, & Predict
Run the main modeling pipeline. This script cleans the test set, trains the XGBoost models on the full training set, and generates the final recommendations for the test set:
        python model_pipeline.py
    (Outputs predictions.csv containing predictions, confidence scores, and action recommendations)


