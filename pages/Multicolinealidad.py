import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Set seed and create data
np.random.seed(23)
X = np.linspace(0, 10, 100)#.reshape(-1, 1)  # Reshape to 2D array
y = 10 * np.exp(-0.2 * (X - 1)**2) - 5 * np.exp(-0.5 * (X - 6)**2) + np.random.normal(0, 1, X.shape)

print(f"X : {X}",f"Y : {y}",sep="\n")

# Visualize the data
plt.plot(X, y, 'b.')
plt.xlabel("X")
plt.ylabel("Y")
st.pyplot(plt)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Create PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=True)
x_train_trans = poly.fit_transform(x_train)
x_test_trans = poly.transform(x_test)

# Define polynomial regression function
def polynomial_regression(degree):
    X_new = np.linspace(0, 10, 100).reshape(-1, 1)
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
        ("poly_features", polybig_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg),
    ])
    polynomial_regression.fit(x_train, y_train)
    y_newbig = polynomial_regression.predict(X_new)
    
    plt.plot(X_new, y_newbig, 'r', label="Degree " + str(degree), linewidth=2)
    plt.plot(x_train, y_train, "b.", linewidth=3)
    plt.plot(x_test, y_test, "g.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    st.pyplot(plt)

# Degree selection in Streamlit
degree = st.selectbox("Select the degree for Polynomial Regression", [i for i in range(1, 26)])

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Generate and split data
np.random.seed(23)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 10 * np.exp(-0.2 * (X - 1)**2) - 5 * np.exp(-0.5 * (X - 6)**2) + np.random.normal(0, 1, X.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

def polynomial_regression(degree):
    X_new = np.linspace(0, 10, 100).reshape(-1, 1)
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    
    # Crear pipeline
    polynomial_regression = Pipeline([
        ("poly_features", polybig_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg),
    ])
    
    # Ajustar el modelo
    polynomial_regression.fit(x_train, y_train)
    y_newbig = polynomial_regression.predict(X_new)
    
    # Graficar resultados
    plt.plot(X_new, y_newbig, 'r', label="Degree " + str(degree), linewidth=2)
    plt.plot(x_train, y_train, "b.", linewidth=3)
    plt.plot(x_test, y_test, "g.", linewidth=3)
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("y")
    st.pyplot(plt)
    
    # Calcular y graficar la matriz de correlación (solo mitad inferior)
    x_train_transformed = polybig_features.fit_transform(x_train)
    data_with_y = np.hstack([x_train_transformed, y_train])
    df = pd.DataFrame(data_with_y, columns=[f"$X^{{{i}}}$" for i in range(1, degree + 1)] + ["y"])
    correlation_matrix = df.corr()
    
    # Crear máscara para la mitad superior
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Graficar matriz de correlación con la máscara
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Matrix (Lower Triangle)")
    st.pyplot(plt)


polynomial_regression(degree)



