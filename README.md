# Water Quality Estimator

This project leverages machine learning to predict the potability of water based on its chemical properties. It includes data cleaning, exploratory analysis, model training, evaluation, and deployment of the best model for real-world use.

## Features
- **Data Preprocessing:** Handles missing values, removes outliers, and prepares features for modeling.
- **Exploratory Data Analysis:** Visualizes feature relationships and class balance using heatmaps, boxplots, pairplots, and count plots.
- **Model Training & Comparison:** Compares Logistic Regression, Random Forest, and XGBoost classifiers to find the best model.
- **Evaluation:** Uses accuracy, F1 score, ROC-AUC, classification reports, and confusion matrices to assess model performance.
- **Deployment:** Saves the best model (Random Forest) for use in web applications.

## Usage
1. Clone this repository and install the required Python packages (see requirements.txt).
2. Run the Jupyter notebook `Water Quality Estimator.ipynb` to explore the data, train models, and evaluate results.
3. Use the provided Streamlit or Flask app to make predictions on new water samples.

## Dataset
The dataset contains water quality measurements such as pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Trihalomethanes, Turbidity, and a target label for potability.

## Results
- Random Forest and XGBoost achieved the best performance.
- The trained Random Forest model is saved as `water_quality_rf_model.pkl` for deployment.

## Applications
- Quick assessment of water potability for individuals or organizations.
- Educational resource for data science and environmental studies.
- Foundation for further research or integration into water monitoring systems.

## Authors
- Piyush Thakur
- Ankush Patial

---
Feel free to use, modify, and contribute to this project!
