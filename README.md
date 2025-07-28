# Churn Prediction Project

## Project Description
This project focuses on predicting customer churn using machine learning techniques. The dataset contains user activity data, and the goal is to classify whether a user will enroll (churn) or not based on their behavior and usage patterns.

## Dataset
The dataset used in this project is loaded from the `archive/appdata10.csv` file. It contains user activity features such as age, number of screens visited, and timestamps of user actions. Additional data about top screens is loaded from `archive/top_screens.csv`.

## Data Preprocessing
- Extracted numerical features and cleaned the dataset by removing irrelevant columns.
- Converted time-related features to appropriate datetime formats.
- Analyzed correlations between features and the target variable.
- Visualized distributions and relationships using histograms, KDE plots, scatter plots, and heatmaps.

## Machine Learning Models
The following models were trained and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Hyperparameter tuning was performed using both Grid Search and Randomized Search with cross-validation to optimize model performance.

## Evaluation Metrics
Models were evaluated using:
- Confusion Matrix visualization
- Accuracy
- Precision
- Recall
- F1 Score
- K-Fold Cross Validation

## How to Run
1. Ensure you have Python 3.x installed.
2. Install required packages:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn python-dateutil
   ```
3. Run the Jupyter notebook `churn prediction.ipynb` to reproduce the analysis and model training.

## Project Structure
- `churn prediction.ipynb`: Main Jupyter notebook containing data analysis and model training.
- `archive/appdata10.csv`: Dataset file.
- `archive/top_screens.csv`: Additional data on top screens.

## Author
This project was created for churn prediction analysis and modeling.
