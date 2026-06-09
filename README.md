# End-to-End Bank Loan Propensity Prediction & MLOps Deployment on AWS

## Project Summary

This project develops and deploys a production-grade machine learning system that identifies bank customers most likely to accept loan offers.

The solution helps retail banks improve marketing efficiency by targeting high-propensity customers instead of conducting broad, low-conversion campaigns.

The project covers the complete machine learning lifecycle, from data preparation and predictive modeling to cloud deployment on AWS using Kubernetes and CI/CD automation.

---

## Business Problem

A retail bank wants to increase loan conversion rates while reducing marketing costs.

Historically, marketing campaigns targeted a large customer base and achieved only single-digit conversion rates.

The objective is to predict which customers are most likely to accept a loan offer so that marketing resources can be focused on high-probability prospects.

### Expected Business Benefits

* Improve loan conversion rates
* Increase marketing efficiency
* Reduce customer acquisition costs
* Improve marketing ROI
* Expand the borrower portfolio

---

## Solution Architecture

```text
Customer Data
      ↓
Data Validation
      ↓
Feature Engineering Pipeline
      ↓
Hist Gradient Boosting Model
      ↓
Probability Scoring
      ↓
Borrower Likelihood Ranking
      ↓
Marketing Campaign System
```

---

## Dataset Overview

| Metric          | Value      |
| --------------- | ---------- |
| Initial Records | 5,000      |
| Final Records   | 4,980      |
| Features        | 14         |
| Target Variable | LoanOnCard |

### Target Variable

| Value | Description                   |
| ----- | ----------------------------- |
| 0     | Customer does not have a loan |
| 1     | Customer has a loan           |

---

## Machine Learning Workflow

```text
Raw Data
    ↓
Data Understanding
    ↓
Data Cleaning
    ↓
EDA & Statistical Analysis
    ↓
Feature Engineering
    ↓
Class Imbalance Handling
    ↓
Model Development
    ↓
Hyperparameter Tuning
    ↓
Model Selection
    ↓
Model Deployment
```

---

## Key Exploratory Findings

### Strong Predictors

* HighestSpend
* MonthlyAverageSpend
* FixedDepositAccount
* Mortgage
* Level

### Multicollinearity

Severe multicollinearity detected between:

* Age
* CustomerSince

Model-specific experiments were conducted to determine the optimal feature selection strategy.

### Class Imbalance

| Class   | Percentage |
| ------- | ---------- |
| No Loan | 90.36%     |
| Loan    | 9.64%      |

The project evaluated:

* Weighted Models
* SMOTE
* SMOTE + Undersampling

---

## Models Evaluated

### Classification Algorithms

* Logistic Regression
* Weighted Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)
* Decision Tree
* Random Forest
* Hist Gradient Boosting
* AdaBoost

### Additional Experiments

* Log Feature Transformation
* Feature Selection
* SMOTE
* Hybrid Resampling
* Hyperparameter Optimization

A total of 20+ model variants were evaluated.

---

## Final Model Comparison

| Model                             | ROC-AUC | Precision | Recall | F1 Score |
| --------------------------------- | ------- | --------- | ------ | -------- |
| Hist Gradient Boosting (Tuned)    | 99.93%  | 97.87%    | 95.83% | 96.84%   |
| Random Forest                     | 99.91%  | 98.90%    | 93.75% | 96.26%   |
| Random Forest + Hybrid Resampling | 99.90%  | 92.23%    | 98.96% | 95.48%   |

---

## Production Model

### Hist Gradient Boosting (Tuned)

Selected because it achieved:

* Highest F1 Score
* Highest ROC-AUC
* Excellent Precision
* Excellent Recall
* Strong Generalization Performance
* Minimal Overfitting

### Final Performance

| Metric    | Value  |
| --------- | ------ |
| ROC-AUC   | 99.93% |
| Precision | 97.87% |
| Recall    | 95.83% |
| F1 Score  | 96.84% |
| Accuracy  | 99.40% |

---

## Feature Importance

### Top Predictive Features

| Rank | Feature             |
| ---- | ------------------- |
| 1    | HighestSpend        |
| 2    | Level               |
| 3    | HiddenScore         |
| 4    | MonthlyAverageSpend |
| 5    | FixedDepositAccount |

---

## Business Impact

The solution enables the bank to:

* Prioritize high-probability borrowers
* Reduce marketing spend on low-conversion prospects
* Increase loan adoption rates
* Improve campaign targeting
* Support data-driven lending strategies

---

## MLOps Architecture

```text
GitHub
   ↓
AWS CodePipeline
   ↓
AWS CodeBuild
   ↓
Docker Image Build
   ↓
Amazon ECR
   ↓
Amazon EKS
   ↓
Flask API
   ↓
Loan Predictions
```

---

## Deployment Architecture

```text
Customer Data
       ↓
Feature Engineering Pipeline
       ↓
Hist Gradient Boosting Model
       ↓
Prediction API (Flask)
       ↓
Docker Container
       ↓
Amazon EKS
       ↓
Marketing Dashboard
```

---

## Technology Stack

### Data Science

* Python
* Pandas
* NumPy
* Scikit-Learn
* SciPy
* Imbalanced-Learn
* Matplotlib
* Seaborn

### MLOps

* Flask
* Docker
* Kubernetes
* Amazon EKS
* Amazon ECR
* AWS CodeBuild
* AWS CodePipeline
* Terraform

### Version Control

* Git
* GitHub

---

## Skills Demonstrated

### Machine Learning

* Classification Modeling
* Feature Engineering
* Statistical Testing
* Class Imbalance Handling
* Hyperparameter Optimization
* Ensemble Learning
* Model Evaluation

### MLOps

* Flask API Development
* Docker Containerization
* Kubernetes Orchestration
* CI/CD Pipelines
* Infrastructure as Code
* AWS Cloud Deployment

---

## Repository Structure

```text
bank-loan-propensity-mlops/
│
├── data/
├── notebooks/
├── src/
├── flask_app/
├── docker/
├── kubernetes/
├── terraform/
├── screenshots/
├── requirements.txt
└── README.md
```

---

## Conclusion

This project demonstrates a complete production-grade machine learning workflow, from business problem definition and statistical analysis to AWS-based MLOps deployment.

The final Hist Gradient Boosting model achieved a ROC-AUC of 99.93% and an F1 Score of 96.84%, enabling highly targeted loan marketing campaigns while maintaining excellent precision and recall.
