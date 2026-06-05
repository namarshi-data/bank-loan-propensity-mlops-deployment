# Bank Loan Propensity Model Deployment using MLOps
Production-grade MLOps project that predicts high-propensity loan customers using Random Forest and deploys the model with Flask, Docker, Kubernetes, AWS EKS, Terraform, and CI/CD pipelines.


## Keyword
Machine Learning, MLOps, Flask, Docker, AWS EKS, Kubernetes, Amazon ECR, CodeBuild, CodePipeline, Terraform, CI/CD, Random Forest, SMOTE, Classification, Banking Analytics, Propensity Modeling, Model Deployment


## Executive Summary
# End-to-End Bank Loan Propensity Prediction using Machine Learning and AWS MLOps

This project develops, evaluates, and deploys a machine learning solution to identify customers most likely to accept loan offers from a retail bank.

Starting from raw customer data, the project performs exploratory data analysis, statistical testing, feature engineering, class imbalance handling, model development, hyperparameter tuning, and model evaluation. Multiple machine learning algorithms were compared, including Logistic Regression, Naive Bayes, Support Vector Machine, Decision Tree, Hist Gradient Boosting, and Random Forest.

The final Random Forest model trained on SMOTE-balanced data achieved a ROC-AUC of 0.999, Recall of 97.9%, Precision of 97.9%, F1-Score of 97.9%, and Accuracy of 99.6%.

To demonstrate production readiness, the selected model is packaged using Flask, containerized with Docker, deployed on AWS Elastic Kubernetes Service (EKS), and automated through CI/CD pipelines using AWS CodeBuild, CodePipeline, and Terraform.


## Deployment Architecture

The trained model is served through a Flask API, containerized using Docker, pushed to Amazon ECR, and deployed to Amazon EKS using Kubernetes manifests. CI/CD is handled using AWS CodeBuild and CodePipeline, while Terraform supports infrastructure provisioning.


## Business Problem

Bank XYZ's customer portfolio is primarily composed of liability customers (depositors) rather than asset customers (borrowers).

Previous marketing campaigns generated only single-digit conversion rates, creating a need for a more targeted and data-driven customer acquisition strategy.

The objective of this project is to identify customers with a high propensity to accept loan offers, enabling focused digital marketing campaigns that improve conversion rates while maintaining existing marketing budgets.


## Dataset Overview

The dataset consists of 5,000 customer records merged from two source files containing demographic, spending, and banking relationship information.

### Target Variable

- LoanOnCard

### Features

- Age
- CustomerSince
- HighestSpend
- HiddenScore
- MonthlyAverageSpend
- Level
- Mortgage
- Security
- FixedDepositAccount
- InternetBanking
- CreditCard


## Project Workflow

Raw Data
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
Model Comparison
↓
Final Model Selection
↓
Model Serialization
↓
Flask API
↓
Docker Containerization
↓
AWS Deployment (EKS)
↓
CI/CD Automation


## Exploratory Data Analysis

### Key Findings

- Target variable was highly imbalanced:
  - No Loan: 90.4%
  - Loan: 9.6%

- HighestSpend, MonthlyAverageSpend, and Mortgage exhibited right-skewed distributions.

- Spending-related variables demonstrated the strongest relationship with loan adoption.

- Age and CustomerSince showed strong multicollinearity.


## Statistical Analysis

The following statistical techniques were applied:

### Numerical Variables

- Mann–Whitney U Test

### Categorical Variables

- Chi-Square Test of Independence

### Correlation Analysis

- Pearson Correlation Matrix

### Multicollinearity Assessment

- Variance Inflation Factor (VIF)


  ### Significant Variables

- HighestSpend
- MonthlyAverageSpend
- Mortgage
- HiddenScore
- Level
- FixedDepositAccount


  ## Feature Engineering

### Data Cleaning

- Removed ID and ZipCode
- Corrected invalid values
- Checked missing values and duplicates

### Transformations

Log transformation applied to:

- HighestSpend
- MonthlyAverageSpend
- Mortgage

### Class Imbalance Handling

- Weighted Models
- SMOTE


  ## Machine Learning Models Evaluated

- Logistic Regression
- Weighted Logistic Regression
- Naive Bayes
- Support Vector Machine (Tuned)
- Decision Tree (Tuned)
- Hist Gradient Boosting
- Random Forest + SMOTE


  ## Model Performance Comparison
| Model No. | Model | ROC-AUC | Recall | Precision | F1-Score | Accuracy |
|------------|--------|---------:|---------:|---------:|---------:|---------:|
| 1 | Logistic Regression - Log Features | 0.972 | 0.625 | 0.857 | 0.723 | 0.954 |
| 2 | Weighted Logistic Regression | 0.971 | 0.917 | 0.442 | 0.597 | 0.881 |
| 3 | Naive Bayes (60:40 Prior) | 0.939 | 0.823 | 0.462 | 0.592 | 0.891 |
| 4 | SVM Tuned | 0.994 | 0.958 | 0.807 | 0.876 | 0.974 |
| 5 | Decision Tree Tuned | 0.955 | 0.917 | 0.936 | 0.926 | 0.986 |
| 6 | Hist Gradient Boosting | 0.999 | 0.948 | 0.989 | 0.968 | 0.994 |
| 7 | Random Forest + SMOTE | 0.999 | 0.979 | 0.979 | 0.979 | 0.996 |


## Final Model Selection

### Random Forest + SMOTE

The Random Forest model trained on SMOTE-balanced data achieved the strongest overall performance.

| Metric | Score |
|----------|---------:|
| ROC-AUC | 0.999 |
| Recall | 97.9% |
| Precision | 97.9% |
| F1-Score | 97.9% |
| Accuracy | 99.6% |


## Why this model?

The selected model correctly identified 97.9% of actual borrowers while maintaining a 97.9% precision rate.

This balance minimizes missed opportunities and unnecessary marketing expenditure, making the model highly suitable for focused digital marketing campaigns.


## Business Impact

The solution enables Bank XYZ to:

- Increase loan conversion rates through targeted campaigns.
- Reduce marketing costs by focusing on high-probability customers.
- Improve marketing ROI.
- Support data-driven customer acquisition strategies.
- Expand the borrower base efficiently.


  ## MLOps Architecture

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
Loan Propensity Predictions


## Technologies Used

### Data Science

- Python
- Pandas
- NumPy
- Scikit-Learn
- Imbalanced-Learn
- Matplotlib
- Seaborn

### MLOps

- Flask
- Docker
- Kubernetes
- AWS EKS
- Amazon ECR
- AWS CodeBuild
- AWS CodePipeline
- Terraform

### Version Control

- Git
- GitHub


  ## Skills Demonstrated

- Exploratory Data Analysis
- Statistical Testing
- Feature Engineering
- Machine Learning
- Hyperparameter Tuning
- Ensemble Learning
- Class Imbalance Handling (SMOTE)
- Model Evaluation
- Flask API Development
- Docker Containerization
- Kubernetes
- AWS Deployment
- CI/CD Pipelines
- Infrastructure as Code (Terraform)
- MLOps
