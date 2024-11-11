import os

# Paths to datasets
DATASETS = {
    "Salary Data": "C:/Users/Admin/Downloads/Data Mining (Codes and Datasets)/App/Data/Simple Linear Regression/Salary_Data.csv",
    "50_Startups": "C:/Users/Admin/Downloads/Data Mining (Codes and Datasets)/App/Data/Multiple Linear Regression/50_Startups.csv",
    "Wine": "C:/Users/Admin/Downloads/Data Mining (Codes and Datasets)/App/Data/Principal Component Analysis (PCA)/Wine.csv",
}

# Available models and their configurations
MODELS = {
    "Simple Linear Regression": {
        "type": "regression",
        "file": "C:/Users/Admin/Downloads/Data Mining (Codes and Datasets)/App/Models/Simple Linear Regression/simple_linear_regression.pkl"
    },
    "Polynomial Regression": {
        "type": "regression",
        "file": "C:/Users/Admin/Downloads/Data Mining (Codes and Datasets)/App/Models/Multiple Linear Regression/Models/multiple_linear_regression_model.pkl"
    },
    "Principal Component Analysis": {
        "type": "regression",
        "file": "C:/Users/Admin/Downloads/Data Mining (Codes and Datasets)/App/Models/Principal Component Analysis (PCA)/Models/principal_component_regression_analysis.pkl"
    },
    # Add more models as needed
}
