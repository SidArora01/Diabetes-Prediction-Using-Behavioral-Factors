# Diabetes-Prediction-Using-Behavioral-Factors

# Group 07 Project - Part 1 Code

## Overview

This project tries to predict risk of Diabetes/Pre-Diabetes based on an individual's behavioral factors
This is part of a Course Project.

## Dependencies

The notebook relies on several Python libraries for data analysis, visualization, and machine learning:

- **NumPy**: Numerical computing.
- **Pandas**: Data manipulation and analysis.
- **Matplotlib/Seaborn**: Data visualization.
- **Scikit-learn**: Machine learning algorithms.

## Methods Used

### Data Collection
The dataset used is sourced from Kaggle. It is the 2021 BRFSS dataset, which includes health indicators related to diabetes.

- **Dataset Link**: [diabetes_012_health_indicators_BRFSS2021.csv](https://www.kaggle.com/datasets/diabetes_012_health_indicators_BRFSS2021.csv)
- **Original Dataset Documentation**: Detailed documentation for each feature is available from the CDC.
  - **Documentation Link**: [CDC BRFSS 2021 Codebook](https://www.cdc.gov/brfss/annual_data/2021/pdf/codebook21_llcp-v2-508.pdf)

The notebook includes methods to import and preprocess data from various sources. Data collection steps might involve:

- Reading data from CSV files.
- Handling missing values and data cleaning.
- Exploratory data analysis (EDA) to understand the dataset.

### Data Processing

Data processing steps include:

- Feature selection and engineering.
- Data normalization and standardization.
- Splitting the data into training and testing sets.

## Steps Involved

1. **Setup Environment**: Import necessary libraries and set up the environment.
2. **Data Collection**: Load the dataset and perform initial data inspection.
3. **Data Cleaning**: Handle missing values and remove or impute them as necessary.
4. **Exploratory Data Analysis (EDA)**: Visualize data distributions and relationships.
5. **Feature Engineering**: Create new features and select relevant ones.
6. **Data Splitting**: Split the data into training and testing sets.
7. **Model Training**: Train machine learning models using the training data.
8. **Model Evaluation**: Evaluate the performance of the models using the testing data.
9. **Visualization**: Create visualizations to interpret the results and model performance.

## Models Used

The notebook uses various machine learning models, such as:

- **Linear Regression**: For predicting continuous variables.
- **Logistic Regression**: For binary classification problems.
- **Decision Trees and Random Forests**: For both classification and regression tasks.
- **Support Vector Machines (SVM)**: For classification tasks.
- **K-Nearest Neighbors (KNN)**: For classification tasks.

## Evaluation Methods

### Model Evaluation

- **Accuracy**: For classification tasks.
- **Precision, Recall, and F1 Score**: For classification tasks, especially with imbalanced datasets.
- **Mean Absolute Error (MAE) and Mean Squared Error (MSE)**: For regression tasks.
- **Cross-Validation**: To assess model performance and ensure it generalizes well to unseen data.

### Visualization

- **Confusion Matrix**: For evaluating classification model performance.
- **ROC Curve and AUC**: For assessing binary classification models.
- **Residual Plots**: For evaluating regression model performance.
- **Feature Importance Plots**: For interpreting model predictions.

## Usage

To view the project, open the `Main.ipynb` file in a web browser. To run or modify the code, use the original Jupyter Notebook (`.ipynb`) file, which can be opened with Jupyter Notebook or JupyterLab.

## Contributors

- Siddharth Arora

For any questions or further information, please contact the project contributor.

