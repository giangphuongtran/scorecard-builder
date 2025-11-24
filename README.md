# Credit Scoring Risk Models & Decision Engine

A comprehensive credit risk modeling system for predicting loan defaults and implementing automated credit acceptance strategies. This project builds multiple risk and marketing models using both traditional statistical methods (SAS) and modern machine learning techniques (Python).

## 🎯 Project Overview

This project develops a complete credit scoring pipeline that includes:
- **Risk Models**: Probability of Default (PD) models for different loan products
- **Marketing Model**: Response probability model for cross-selling
- **Decision Engine**: Automated rules-based system for credit acceptance
- **Simulation & Profitability Analysis**: Backtesting and profit/loss evaluation over historical periods (1975-1987)

## 📊 Models

### 1. PD Ins Model
**Target**: `default12 = 1` for instalment loans (`product = 'ins'`)  
**Purpose**: Predicts the probability of default within 12 months for instalment loan products.

### 2. PD Css Model
**Target**: `default12 = 1` for cash loans (`product = 'css'`)  
**Purpose**: Predicts the probability of default within 12 months for cash loan products.

### 3. PD Css Cross Model
**Target**: `default_cross12 = 1` for cash loans at the time of applying for instalment loan  
**Purpose**: Estimates default risk for existing cash loan customers when they apply for an instalment loan.

### 4. PR Css Cross Model (Marketing Model)
**Target**: `cross_response = 1` when applying for instalment and/or cash loan  
**Purpose**: Predicts the probability of cross-selling response, enabling marketing campaign optimization.

### 5. Decision Engine
**Purpose**: Implements business rules combining all models to automate credit acceptance decisions with optimized risk-return trade-offs.

### 6. Profitability Analysis
**Period**: 1975-1987  
**Purpose**: Evaluates the financial performance of the credit acceptance strategy through comprehensive profit and loss reporting.

## 🛠️ Technologies

### Data Processing & Modeling
- **SAS**: Traditional credit scoring pipeline, WOE transformation, variable selection, scorecard development
- **Python**: Machine learning models and advanced analytics
  - `pandas`, `numpy` - Data manipulation
  - `scikit-learn` - Traditional ML algorithms
  - `XGBoost` - Gradient boosting models
  - `TensorFlow/Keras` - Neural networks
  - `SHAP` - Model interpretability
  - `statsmodels` - Statistical modeling

### Analysis & Visualization
- **Jupyter Notebooks** - Interactive data analysis and model development
- **Matplotlib/Plotly** - Data visualization
- **Excel** - Business reporting and scorecard presentation

### Tools
- `pyreadstat` / `sas7bdat` - Reading SAS datasets in Python
- `openpyxl`, `XlsxWriter` - Excel file generation

## 📁 Project Structure
TBD

## 🔄 Workflow

1. **Data Preparation**: Create Analysis Base Tables (ABT) with train/validation splits
2. **Feature Engineering**: 
   - Variable binning (nominal and interval)
   - WOE (Weight of Evidence) transformation
   - Variable selection and pre-screening
3. **Model Development**: Build 4 models (PD Ins, PD Css, PD Css Cross, PR Css Cross)
4. **Model Validation**: 
   - Bootstrap validation
   - Cross-validation
   - Gini coefficient evaluation
   - Model assessment reports
5. **Calibration**: Finalize cut-offs and decision rules
6. **Decision Engine**: Implement automated credit acceptance rules
7. **Simulation**: Run historical backtesting (1975-1987)
8. **Reporting**: Generate profit/loss reports and model documentation

## 📈 Model Performance Metrics

- **Gini Coefficient**: Model discrimination ability
- **KS Statistic**: Kolmogorov-Smirnov test for model separation
- **AUC-ROC**: Area Under the ROC Curve
- **Bootstrap Validation**: Robust performance estimates
- **Profit/Loss Metrics**: Business impact evaluation

## 📚 Methodology

The project follows industry-standard credit risk modeling practices:
- **WOE Transformation**: Weight of Evidence for categorical variables
- **Variable Selection**: Statistical significance testing and correlation analysis
- **Model Validation**: Train/validation/test splits with bootstrap resampling
- **Scorecard Development**: Point-based scoring system
- **Decision Rules**: Risk-based acceptance/rejection thresholds
