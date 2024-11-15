import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, RandomOverSampler
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import scipy.stats as stats
from scipy.stats import shapiro, anderson, kstest
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go


problema = st.sidebar.selectbox("Seleccione el problema:", ["Regresion", "Clasificacion"],key="s")

if problema == "Regresion":

    st.markdown("# Regresion Lineal")

    Wine = pd.read_csv("WineQT.csv")
    procesamiento = st.selectbox("Preprocesamiento:", ["Ninguno", "EDA"],key="o")

    Wine = Wine.iloc[:, :-1]

    if procesamiento == "Ninguno":
        fig = go.Figure()
        for col in Wine.columns:
            fig.add_trace(go.Box(y=Wine[col], name=col))
        fig.update_layout(title="Datos crudos", xaxis_title="Características", yaxis_title="Valores")
        st.plotly_chart(fig)

    if procesamiento == "EDA":
        scaler = StandardScaler()
        Wine_imputed = Wine.drop(columns=["quality","fixed acidity", "density", "pH"])
        Wine_scaled = pd.DataFrame(scaler.fit_transform(Wine_imputed), columns=Wine_imputed.columns)
        Wine_scaled['quality'] = Wine['quality']
        Wine = Wine_scaled

        # Gráfico de caja después de la normalización
        fig = go.Figure()
        for col in Wine_scaled.columns[:-1]:
            fig.add_trace(go.Box(y=Wine_scaled[col], name=col))
        fig.update_layout(title="Características después de la normalización", xaxis_title="Características", yaxis_title="Valores")
        st.plotly_chart(fig)

        df = Wine[['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol', 'quality']]
    
    else:
        Wine = pd.read_csv("WineQT.csv")
        Wine = Wine.iloc[:, :-1]
        df = Wine
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Gráfico de caja sin outliers
    fig = go.Figure()
    for col in df.columns[:-1]:
        fig.add_trace(go.Box(y=df[col], name=col))
    fig.update_layout(title="Características sin outliers", xaxis_title="Características", yaxis_title="Valores")
    st.plotly_chart(fig)

    if procesamiento == "EDA":
        X = df[['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'sulphates', 'alcohol']]
        y = df[['quality']]

    else:
        X = df.drop(columns="quality")
        y = df[['quality']]

    if procesamiento == "Ninguno":
        st.code('''
                Wine = pd.read_csv("WineQT.csv")
                Wine = Wine.iloc[:, :-1]
                X = Wine.drop(columns="quality")
                y = Wine[['quality']]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

                ols_model = sm.OLS(y_train, X_train_const)
                ols_results = ols_model.fit()

                summary = ols_results.summary()

                print(summary)
                ''')
    
    else:
        st.code('''
                Wine = pd.read_csv("WineQT.csv")
                Wine = Wine.iloc[:, :-1]
                X = Wine.drop(columns="quality","fixed acidity", "density", "pH")
                y = Wine[['quality']]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

                ols_model = sm.OLS(y_train, X_train_const)
                ols_results = ols_model.fit()

                summary = ols_results.summary()

                print(summary)
                ''')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_train)
    residuos = y_train - y_pred

    X_train_const = sm.add_constant(X_train)  # Agregar constante para el intercepto

    ols_model = sm.OLS(y_train, X_train_const)
    ols_results = ols_model.fit()

    summary = ols_results.summary()

    coef_table = summary.tables[1]  # El índice 1 tiene los coeficientes

    coef_table_df = pd.DataFrame(coef_table)

    # Convertir todas las celdas a cadenas de texto para asegurar compatibilidad
    coef_table_df = coef_table_df.applymap(str)

    # Configurar nombres de columnas de forma explícita
    coef_table_df.columns = ["Variable", "Coef", "std err", "t", "P>|t|", "[0.025", "0.975]"]
    coef_table_df = coef_table_df.drop(0)  # Eliminar fila de encabezado duplicado
    coef_table_df.set_index("Variable", inplace=True)

    # Asegurarse de que los valores en la columna "P>|t|" sean numéricos
    coef_table_df["P>|t|"] = pd.to_numeric(coef_table_df["P>|t|"], errors='coerce')

    # Aplicar color a los valores p en la columna "P>|t|"
    styled_coef_table = coef_table_df.style.applymap(
        lambda val: 'background-color: #B0ED8B' if val < 0.05 else 'background-color: #ED8181' if pd.notnull(val) else '',
        subset=["P>|t|"]
    )

    _,col1,_ = st.columns([0.2,1,0.2])

    st.table(styled_coef_table)

    # Gráfico de residuos vs valores predichos
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred.flatten(), y=residuos.values.flatten(), mode="markers", name="Residuos"))
    fig.add_trace(go.Scatter(x=y_pred.flatten(), y=np.repeat(np.mean(residuos), len(y_pred)), mode="lines", name="Media de Residuos", line=dict(color="red", dash="dash")))
    fig.update_layout(title="Residuos vs Valores Predichos", xaxis_title="Valores Predichos", yaxis_title="Residuos")
    st.plotly_chart(fig)

    # Pruebas de homocedasticidad y normalidad
    exog = sm.add_constant(X_train)
    bp_test = het_breuschpagan(residuos, exog)
    labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
    bp_results = dict(zip(labels, bp_test))
    for key, value in bp_results.items():
        st.write(f"{key}: {value}")

    # Gráfico de residuos vs observaciones
    observaciones = np.arange(len(residuos))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=observaciones, y=residuos.values.flatten(), mode="lines+markers", name="Residuos"))
    fig.add_trace(go.Scatter(x=observaciones, y=np.zeros(len(observaciones)), mode="lines", name="Referencia", line=dict(color="red", dash="dash")))
    fig.update_layout(title="Residuos vs Observaciones", xaxis_title="Observaciones", yaxis_title="Residuos")
    st.plotly_chart(fig)

    # Estadístico de Durbin-Watson
    dw_stat = durbin_watson(residuos)
    st.write("Estadístico de Durbin-Watson:", dw_stat)

    # Q-Q plot
    theoretical_quantiles = np.linspace(0.001, 0.999, len(residuos.values.flatten()))
    theoretical_values = stats.norm.ppf(theoretical_quantiles)
    residuals_sorted = np.sort(residuos.values.flatten())

    slope, intercept = np.polyfit(theoretical_values, residuals_sorted, 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=theoretical_values,
        y=residuals_sorted,
        mode='markers',
        name='Residuos vs Cuartiles teóricos',
        marker=dict(color='lightblue',size = 8)
    ))

    fig.add_trace(go.Scatter(
        x=theoretical_values,
        y=slope * theoretical_values + intercept,
        mode='lines',
        name='Línea de referencia',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title="Gráfico de probabilidad (Q-Q plot)",
        xaxis_title="Cuartiles teóricos",
        yaxis_title="Residuos",
        height = 700,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig)

    residuos_array = residuos.values.flatten()
    # Pruebas de normalidad
    shapiro_stat, shapiro_p = shapiro(residuos_array)
    st.write("Prueba de Shapiro-Wilk:")
    st.write("Estadístico:", shapiro_stat)
    st.write("p-valor:", shapiro_p)

    anderson_result = anderson(residuos_array, dist='norm')
    st.write("Prueba de Anderson-Darling:")
    st.write("Estadístico:", anderson_result.statistic)
    st.write("Valores críticos:")
    for sl, cv in zip(anderson_result.significance_level, anderson_result.critical_values):
        st.write(f"Nivel de significancia: {sl}%, Valor crítico: {cv}")

    if anderson_result.statistic < anderson_result.critical_values[2]:
        st.write("No se rechaza la hipótesis nula de normalidad (al 5% de significancia).")
    else:
        st.write("Se rechaza la hipótesis nula de normalidad (al 5% de significancia).")

    ks_stat, ks_p = kstest(residuos_array, 'norm', args=(np.mean(residuos_array), np.std(residuos_array, ddof=1)))
    st.write("Prueba de Kolmogorov-Smirnov:")
    st.write("Estadístico:", ks_stat)
    st.write("p-valor:", ks_p)

    # Factor de Inflación de la Varianza (VIF)
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X_train.columns
    vif_data['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    st.write("Factor de Inflación de la Varianza (VIF):")
    st.write(vif_data)




    ###### Regresión polinómica
    st.markdown("# Regresión Polinómica")
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear_regression', LinearRegression())
    ])
    param_grid = {'poly_features__degree': [2, 3, 4]}
    kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kfolds, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_degree = grid_search.best_params_['poly_features__degree']
    best_model = grid_search.best_estimator_
    st.write("Mejor grado de polinomio encontrado:", best_degree)

    poly = PolynomialFeatures(degree=best_degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)

    # Agregar la constante
    X_train_poly_const = sm.add_constant(X_train_poly)

    # Ajustar el modelo polinómico en statsmodels
    ols_poly_model = sm.OLS(y_train, X_train_poly_const)
    ols_poly_results = ols_poly_model.fit()

    summary = ols_poly_results.summary()

    coef_table = summary.tables[1]  # El índice 1 tiene los coeficientes

    coef_table_df = pd.DataFrame(coef_table)

    # Convertir todas las celdas a cadenas de texto para asegurar compatibilidad
    coef_table_df = coef_table_df.applymap(str)

    # Configurar nombres de columnas de forma explícita
    coef_table_df.columns = ["Variable", "Coef", "std err", "t", "P>|t|", "[0.025", "0.975]"]
    coef_table_df = coef_table_df.drop(0)  # Eliminar fila de encabezado duplicado
    coef_table_df.set_index("Variable", inplace=True)

    # Asegurarse de que los valores en la columna "P>|t|" sean numéricos
    coef_table_df["P>|t|"] = pd.to_numeric(coef_table_df["P>|t|"], errors='coerce')

    # Aplicar color a los valores p en la columna "P>|t|"
    styled_coef_table = coef_table_df.style.applymap(
        lambda val: 'background-color: #B0ED8B' if val < 0.05 else 'background-color: #ED8181' if pd.notnull(val) else '',
        subset=["P>|t|"]
    )

    _,col1,_ = st.columns([0.2,1,0.2])

    st.table(styled_coef_table)

    y_pred_poly = ols_poly_results.predict(X_train_poly_const)
    rmse_poly = np.sqrt(mean_squared_error(y_train, y_pred_poly))
    r2_poly = r2_score(y_train, y_pred_poly)

    st.write("Resultados de Regresión Polinómica:")
    st.write("RMSE:", rmse_poly)
    st.write("R²:", r2_poly)
    
    y_pred_poly = best_model.predict(X_train)
    rmse_poly = np.sqrt(mean_squared_error(y_train, y_pred_poly))
    r2_poly = r2_score(y_train, y_pred_poly)

    st.write("Resultados de Regresión Polinómica:")
    st.write("RMSE:", rmse_poly)
    st.write("R²:", r2_poly)

    # Gráfico de residuos vs valores predichos para regresión polinómica
    residuos_poly = y_train - y_pred_poly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred_poly.flatten(), y=residuos_poly.values.flatten(), mode="markers", name="Residuos"))
    fig.add_trace(go.Scatter(x=y_pred_poly.flatten(), y=np.repeat(np.mean(residuos_poly), len(y_pred_poly)), mode="lines", name="Media de Residuos", line=dict(color="red", dash="dash")))
    fig.update_layout(title="Residuos vs Valores Predichos (Regresión Polinómica)", xaxis_title="Valores Predichos", yaxis_title="Residuos")
    st.plotly_chart(fig)

    observaciones = np.arange(len(residuos_poly))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=observaciones, y=residuos_poly.values.flatten(), mode="lines+markers", name="Residuos"))
    fig.add_trace(go.Scatter(x=observaciones, y=np.zeros(len(observaciones)), mode="lines", name="Referencia", line=dict(color="red", dash="dash")))
    fig.update_layout(title="Residuos vs Observaciones", xaxis_title="Observaciones", yaxis_title="Residuos")
    st.plotly_chart(fig)

    theoretical_quantiles = np.linspace(0.001, 0.999, len(residuos_poly.values.flatten()))
    theoretical_values = stats.norm.ppf(theoretical_quantiles)
    residuals_sorted = np.sort(residuos_poly.values.flatten())

    slope, intercept = np.polyfit(theoretical_values, residuals_sorted, 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=theoretical_values,
        y=residuals_sorted,
        mode='markers',
        name='Residuos vs Cuartiles teóricos',
        marker=dict(color='lightblue',size = 8)
    ))

    fig.add_trace(go.Scatter(
        x=theoretical_values,
        y=slope * theoretical_values + intercept,
        mode='lines',
        name='Línea de referencia',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title="Gráfico de probabilidad (Q-Q plot)",
        xaxis_title="Cuartiles teóricos",
        yaxis_title="Residuos",
        height = 700,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    st.plotly_chart(fig)

    
elif problema == "Clasificacion":

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, classification_report
    import matplotlib.pyplot as plt

    ###### Clasificacion
    Wine = pd.read_csv("WineQT.csv")

    X_original = Wine.drop(columns=["quality","Id","pH","density"])
    y_original = np.where(Wine.quality >= 7,1,0)

    adasyn = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_original, y_original)

    # Borderline-SMOTE
    borderline_smote = BorderlineSMOTE(sampling_strategy='minority', random_state=42, kind='borderline-1')
    X_borderline, y_borderline = borderline_smote.fit_resample(X_original, y_original)

    ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
    X_ros, y_ros = ros.fit_resample(X_original, y_original)

    # Submuestreo simple con RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    X_rus, y_rus = rus.fit_resample(X_original, y_original)

    # SMOTE
    smote = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=5)
    X_smote, y_smote = smote.fit_resample(X_original, y_original)

    balance = st.selectbox("Selecciona tecnica de balance:",
                        ["Original","ADASYN","BORDERLINE_SMOTE","RANDOM OVER SAMPLING", "RANDOM UNDER SAMPLING"])

    if balance == "Original":
        X_train, X_test, y_train, y_test = train_test_split(X_original, y_original, test_size=0.2, random_state=42)

    elif balance == "ADASYN":
        X_train, X_test, y_train, y_test = train_test_split(X_adasyn, y_adasyn, test_size=0.2, random_state=42)

    elif balance == "BORDERLINE_SMOTE":
        X_train, X_test, y_train, y_test = train_test_split(X_borderline, y_borderline, test_size=0.2, random_state=42)

    elif balance == "RANDOM OVER SAMPLING":
        X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size=0.2, random_state=42)

    elif balance == "RANDOM UNDER SAMPLING":
        X_train, X_test, y_train, y_test = train_test_split(X_rus, y_rus, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance'], 'p': [1, 2]}  #Manhattan,  Euclidiana
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_
    st.write("Mejores parámetros:", grid_search.best_params_)

    y_pred_knn = best_knn.predict(X_test)
    y_pred_proba_knn = best_knn.predict_proba(X_test)[:, 1]

    X_scaled = sm.add_constant(X_scaled)

    # Crear el modelo en statsmodels
    log_reg_sm = sm.Logit(y_train, X_scaled).fit()

    summary = log_reg_sm.summary()

    coef_table = summary.tables[1]  # El índice 1 tiene los coeficientes

    coef_table_df = pd.DataFrame(coef_table)

    # Convertir todas las celdas a cadenas de texto para asegurar compatibilidad
    coef_table_df = coef_table_df.applymap(str)

    # Configurar nombres de columnas de forma explícita
    coef_table_df.columns = ["Variable", "Coef", "std err", "t", "P>|t|", "[0.025", "0.975]"]
    coef_table_df = coef_table_df.drop(0)  # Eliminar fila de encabezado duplicado
    coef_table_df.set_index("Variable", inplace=True)

    # Asegurarse de que los valores en la columna "P>|t|" sean numéricos
    coef_table_df["P>|t|"] = pd.to_numeric(coef_table_df["P>|t|"], errors='coerce')

    # Aplicar color a los valores p en la columna "P>|t|"
    styled_coef_table = coef_table_df.style.applymap(
        lambda val: 'background-color: #B0ED8B' if val < 0.05 else 'background-color: #ED8181' if pd.notnull(val) else '',
        subset=["P>|t|"]
    )

    st.table(styled_coef_table)

    # Establecer un umbral ajustable para la predicción
    threshold = st.slider("Selecciona el umbral de predicción:", 0.0, 1.0, 0.5, 0.01)

    # Predicciones de probabilidad para el conjunto de entrenamiento y prueba
    y_pred_train_proba = log_reg_sm.predict(X_scaled)
    y_pred_test_proba = log_reg_sm.predict(sm.add_constant(scaler.transform(X_test), has_constant='add'))

    # Convertir probabilidades en etiquetas según el umbral seleccionado
    y_pred_train = (y_pred_train_proba >= threshold).astype(int)
    y_pred_test = (y_pred_test_proba >= threshold).astype(int)

    # Matrices de confusión para entrenamiento y prueba
    conf_matrix_train = confusion_matrix(y_train, y_pred_train)
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Matriz de Confusión - Conjunto de Entrenamiento")
        fig_train = go.Figure(data=go.Heatmap(
            z=conf_matrix_train,
            x=["Predicho 0", "Predicho 1"],
            y=["Real 0", "Real 1"],
            colorscale="Blues",
            hoverongaps=False
        ))
        fig_train.update_layout(title="Matriz de Confusión (Entrenamiento)", xaxis_title="Predicción", yaxis_title="Real")
        st.plotly_chart(fig_train)

    with col2:
        st.write("Matriz de Confusión - Conjunto de Prueba")
        fig_test = go.Figure(data=go.Heatmap(
            z=conf_matrix_test,
            x=["Predicho 0", "Predicho 1"],
            y=["Real 0", "Real 1"],
            colorscale="Blues",
            hoverongaps=False
        ))
        fig_test.update_layout(title="Matriz de Confusión (Prueba)", xaxis_title="Predicción", yaxis_title="Real")
        st.plotly_chart(fig_test)

    # Gráficos ROC y de Precisión-Recall
    precision, recall, thresholds_pr = precision_recall_curve(y_train, y_pred_train_proba)
    pr_auc = auc(recall, precision)

    fpr, tpr, thresholds_roc = roc_curve(y_train, y_pred_train_proba)
    roc_auc = auc(fpr, tpr)

    # Encontrar el punto en las curvas correspondiente al umbral seleccionado
    closest_idx_pr = np.argmin(np.abs(thresholds_pr - threshold))
    closest_idx_roc = np.argmin(np.abs(thresholds_roc - threshold))

    fig_combined = make_subplots(rows=1, cols=2, subplot_titles=("Curva ROC", "Curva de Precisión-Recall"))

    # Curva ROC
    fig_combined.add_trace(
        go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})', line=dict(color='darkorange')),
        row=1, col=1
    )
    fig_combined.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Línea diagonal', line=dict(color='navy', dash='dash')),
        row=1, col=1
    )
    fig_combined.add_trace(
        go.Scatter(x=[fpr[closest_idx_roc]], y=[tpr[closest_idx_roc]], mode='markers',
                name=f'Umbral = {threshold:.2f}', marker=dict(color='red', size=10)),
        row=1, col=1
    )

    # Curva de Precisión-Recall
    fig_combined.add_trace(
        go.Scatter(x=recall, y=precision, mode='lines', name=f'Precision-Recall curve (AUC = {pr_auc:.2f})', line=dict(color='blue')),
        row=1, col=2
    )
    fig_combined.add_trace(
        go.Scatter(x=[recall[closest_idx_pr]], y=[precision[closest_idx_pr]], mode='markers',
                name=f'Umbral = {threshold:.2f}', marker=dict(color='red', size=10)),
        row=1, col=2
    )

    # Configuración de la figura
    fig_combined.update_layout(
        title="Curvas ROC y de Precisión-Recall",
        showlegend=True,
        legend = dict(
        orientation="h",
        yanchor="top",
        y=-0.2,
        xanchor="center",
        x=0.5
        )
    )
    fig_combined.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig_combined.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig_combined.update_xaxes(title_text="Recall", row=1, col=2)
    fig_combined.update_yaxes(title_text="Precision", row=1, col=2)

    # Mostrar la figura combinada
    st.plotly_chart(fig_combined)