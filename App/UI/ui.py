import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
from config.model_config import DATASETS, MODELS
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_data(dataset_name):
    data_path = DATASETS.get(dataset_name)
    if data_path:
        data = pd.read_csv(data_path)
        return data
    st.error("Dataset not found.")
    return None

def display_data(dataset_name):
    data = load_data(dataset_name)
    if data is not None:
        st.write("### Dataset")
        st.write(data.head())
    return data


# Assuming MODELS is a predefined dictionary with model paths
def build_model(model_name, dataset_name):
    # Cargar los datos según el dataset_name
    data = load_data(dataset_name)
    
    if data is not None:
        # Lógica específica para cada dataset y modelo
        if dataset_name == "Salary Data":  # Ejemplo: Dataset con 'YearsExperience' y 'Salary'
            if 'YearsExperience' in data.columns and 'Salary' in data.columns:
                X = data[['YearsExperience']]  # Solo 'YearsExperience' como predictor
                y = data['Salary']
            else:
                st.error(f"Dataset '{dataset_name}' no tiene las columnas esperadas.")
                return
            
        elif dataset_name == "50_Startups":  # Ejemplo: Dataset con múltiples características
            # Verificar que todas las columnas necesarias existen
            if 'R&D Spend' in data.columns and 'Administration' in data.columns and 'Marketing Spend' in data.columns and 'State' in data.columns and 'Profit' in data.columns:
                # Seleccionar las características (X) y la variable objetivo (y)
                X = dataset_name[['R&D Spend','Administration','Marketing Spend','State']]
                y = dataset_name['Profit']
                
                # Crear un pipeline para manejar las columnas categóricas y numéricas
                # Usamos OneHotEncoder para la columna 'State' (categórica) y SimpleImputer para manejar valores faltantes
                numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
                categorical_features = ['State']

                # Crear un transformador de columnas
                numeric_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='mean'))  # Imputar valores faltantes con la media
                ])
                
                categorical_transformer = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputar valores faltantes
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHotEncoding para la columna categórica
                ])

                # Crear el transformador compuesto
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numeric_transformer, numeric_features),
                        ('cat', categorical_transformer, categorical_features)
                    ]
                )
                
                # Crear un modelo completo que incluya tanto el preprocesamiento como el modelo de regresión
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', LinearRegression())  # Usamos regresión lineal
                ])
                
            
        elif dataset_name == "Wine":  # Otro tipo de dataset, con columnas diferentes
            if 'Alcohol' in data.columns and 'Customer_Segment' in data.columns:
                X = dataset_name.iloc[:, :-1].values
                y = dataset_name.iloc[:, -1].values
            else:
                st.error(f"Dataset '{dataset_name}' no tiene las columnas esperadas.")
                return
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        
        # Diccionario de modelos
        model_dict = {
            "Simple Linear Regression": LinearRegression(),
            "Multiple Linear Regression": LinearRegression(),
            "Principal Component Analysis": LinearRegression(),
            "Lasso Regression": Lasso(),
            "Ridge Regression": Ridge(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }
        
        # Seleccionar el modelo en función de model_name
        model = model_dict.get(model_name)
        
        if model is None:
            st.error(f"Model '{model_name}' is not supported.")
            return
        
        # Entrenar el modelo
        model.fit(X_train, y_train)
        
        # Guardar el modelo
        model_path = MODELS[model_name]["file"]
        joblib.dump(model, model_path)
        
        st.write(f"### {model_name} trained successfully!")
        st.write(f"Model Score: {model.score(X_test, y_test):.2f}")

def predict_salary(model_name, years_of_experience):
    model_path = MODELS[model_name]["file"]
    model = joblib.load(model_path)
    prediction = model.predict([[years_of_experience]])
    return prediction[0]

def show_prediction(model_name):
    st.write("### Salary Prediction")
    years_of_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
    if st.button("Predict"):
        salary = predict_salary(model_name, years_of_experience)
        st.write(f"Predicted Salary: ${salary:.2f}")

# Configuración en Streamlit para seleccionar el modelo
st.sidebar.title("Select Model")
model_name = st.sidebar.selectbox("Choose a model", list(MODELS.keys()))
dataset_name = "Salary Data"  # Puedes hacer que este sea dinámico si tienes más datasets disponibles

# Mostrar datos y entrenar el modelo
st.title("Multiple Model Regression App")
display_data(dataset_name)
if st.sidebar.button("Train Model"):
    build_model(model_name, dataset_name)

# Mostrar la predicción de salario
show_prediction(model_name)
