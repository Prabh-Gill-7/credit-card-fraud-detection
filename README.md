# Credit Card Fraud Detection

## Proejct overview
This repository hosts an end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions. Utilizing real-world anonymized transaction data, the system addresses severe class imbalances to identify suspicious behaviors with high precision and sensitivity.

## Problem Statement
Credit card fraud costs financial institutions and consumers billions of dollars annually. The primary technical hurdle in developing an automated detection system is the extreme scarcity of fraud cases. In typical transaction streams, fraudulent actions make up less than 0.2% of all records. Traditional Machine Learning models often default to predicting everything as legitimate, completely failing to detect actual anomalies.

## Objective
* Develop a robust classification model capable of isolating highly imbalanced fraudulent transactions.
* Maximize the **Recall** and **Area Under the Precision-Recall Curve (AUPRC)** to capture as many fraud cases as possible while minimizing false alarms for legitimate users.
* Create a modular, deployment-ready code architecture optimized for seamless integration with real-time prediction environments.

## Solution with flowchart(Architecture)
The application handles processing through a distinct, structured multi-stage execution pipeline:

```text
[ Raw CSV / Live Input ]
         │
         ▼
┌────────────────────────────────────────┐
│ Data Preprocessing & Scaling           │ -> (StandardScaler handles Time/Amount)
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Resampling & Class Imbalance Fix      │ -> (SMOTE / Balanced Class Weights)
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Model Training & Evaluation            │ -> (Random Forest / Ensemble Classifier)
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ Model Validation & Output Generation    │ -> (Saves Confusion Matrix, ROC, Metrics)
└────────────────────────────────────────┘
```

## Key Feature
* **Imbalance Management:** Implements robust oversampling and algorithmic weighting adjustments to handle skewed datasets safely.
* **Modular Pipeline Architecture:** Separate dedicated scripts manage mock data streaming, core model training, and distinct inference testing.
* **Comprehensive Performance Visualizations:** Automated generation of Confusion Matrices, ROC curves, and Feature Importance graphs during every cycle.

## Technology Stack
* **Programming Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning Library:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn

## Project Structure
```text
├── class_distribution.png   # Visualization highlighting extreme target skewness
├── amount_distribution.png  # Analytical plot outlining transaction value spreads
├── confusion_matrix.png     # Validation performance breakout matrix
├── roc_curve.png            # Visual depiction of True Positive vs False Positive rates
├── feature_importance.png   # Graphical ranking of critical decision variables
├── fraud_detection.py       # Main operational script handling preprocessing, training & evaluation
├── generate_sample_data.py  # Utility file building realistic test cases for evaluation
├── test_prediction.py       # Test script verifying pipeline behavior against isolated samples
├── requirements.txt         # Text registry mapping precise dependency package configurations
└── README.md                # Comprehensive project documentation
```

## Project Demonstares
The system maps out decision traits via structured graphical distributions and model analytics. 

### Visual Performance Graphs
Below are the results generated from training and validating the model:

#### 1. Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

#### 2. ROC Curve
![ROC Curve](roc_curve.png)

#### 3. Feature Importance
![Feature Importance](feature_importance.png)

## Project Feature Status
- [x] Data ingestion and Feature Scaling
- [x] Class Imbalance handling implementation
- [x] Model Training Pipeline configuration
- [x] Automated Performance Chart Generation
- [x] Isolated Inference Test Environment setup
- [ ] Real-time API Endpoint Development (Planned)

## Developed By
* **Prabhdeep Singh** - [GitHub Profile](https://github.com)
* *Dataset Source Acknowledgement:* Public reference data provided by [Kaggle](https://kaggle.com).

## Learning Outcomes
* **Handling Massive Data Skew:** Mastered real-world adjustment strategies for training models when target vectors represent <1% of ambient inputs.
* **Advanced Evaluation Metrics:** Moved away from baseline accuracy metrics toward granular evaluation via Precision, Recall, F1-Scores, and Precision-Recall Curves.
* **Production Coding Practices:** Developed a deep understanding of constructing modular, multi-file codebases that segregate mock generation, core pipelines, and runtime sanity testing.

## Conclusion
This implementation demonstrates that standard supervised ensemble architectures, when augmented with precise scaling and algorithmic balance corrections, can accurately predict anomalous transactions without flooding system operators with false flags. The modular code layout ensures that this pipeline can be adapted or moved into distributed streaming data frameworks with ease.