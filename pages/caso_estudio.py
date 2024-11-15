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
        st.write('''<p style='font-size:23px;'>
                En esta parte se realiza la regresion lineal y polinomica sin tener en cuenta ningun procesamiento
                mas alla de quitar los outliers para evaluar como es el rendimiento del modelo con los datos crudos y sin
                meterle mucha ciencia al procedimiento, en la seccion de EDA, se detallara un procesamiento mas amplio
                </p>''',unsafe_allow_html=True)

        fig = go.Figure()
        for col in Wine.columns:
            fig.add_trace(go.Box(y=Wine[col], name=col))
        fig.update_layout(title="Datos crudos", xaxis_title="Características", yaxis_title="Valores")
        st.plotly_chart(fig)

    if procesamiento == "EDA":
        st.write('''<p style='font-size:23px;'>
                En esta parte se realiza la regresion lineal y polinomica teniendo en cuenta un procesamiento el cual incluye un 
                entendimiento de las variables para quitar posibles redundancias, quitar los outliers y normalizar las variables para evaluar 
                la mejoria en el rendimiento del modelo con los datos procesados y entendidos:
                </p>''',unsafe_allow_html=True)
        
        st.write("""
        <p style='font-size:23px;'>El conjunto de datos contiene las siguientes características:

        - **fixed acidity**: Acidez fija del vino.
        - **volatile acidity**: Acidez volátil del vino.
        - **citric acid**: Contenido de ácido cítrico.
        - **residual sugar**: Azúcar residual.
        - **chlorides**: Concentración de cloruros.
        - **free sulfur dioxide**: Dióxido de azufre libre.
        - **total sulfur dioxide**: Dióxido de azufre total.
        - **density**: Densidad del vino.
        - **pH**: Valor de pH.
        - **sulphates**: Concentración de sulfatos.
        - **alcohol**: Contenido alcohólico del vino.
        - **quality**: Clasificación de la calidad del vino (escala del 0 al 10).
        </p>""",unsafe_allow_html=True)

        st.write('''
        <p style='font-size:23px;'><b>FORMA INTUITVA</b><br>
        La densidad de un vino está muy influenciada por componentes como el contenido de alcohol y el azúcar residual. Estos factores afectan directamente la masa y, por ende, la densidad del líquido. Como estos atributos ya están representados, density se vuelve redundante.
        <br><br>El pH, que mide la acidez, es otro aspecto que resulta de la combinación de distintas sustancias ácidas en el vino, como el ácido cítrico (citric acid) y la acidez volátil (volatile acidity). Como estos componentes específicos están directamente cuantificados en otras variables, el pH puede ser considerado como una medida general que añade poca información adicional.
        <br><br>La acidez fija (fixed acidity) también se ve reflejada en variables como citric acid y volatile acidity. Estas variables representan distintos tipos de ácidos específicos, por lo que contar con una medida general de acidez fija es redundante, ya que puede descomponerse en estos elementos individuales.
        </p>''',unsafe_allow_html=True)

        st.write('''
        <p style='font-size:23px;'><b>MULTICOLINEALIDAD</b><br>
        Las variables density, pH y fixed acidity mostraron los valores de VIF más elevados en el modelo de regresion que se hizo 
        anteriormente, lo que sugiere que están siendo explicadas por otras variables presentes en el conjunto de datos que como
        anteriormente desde un punto de vista intuitivo, era el caso 
        </p>''',unsafe_allow_html=True)

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

    st.write("## **Coeficientes estimados**")

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

    with col1:
        st.table(styled_coef_table)
        st.write(f'''
            <p style='font-size:20px;'> Estos fueron los coeficientes estimados por el modelo con su respeciva significancia que
                 nos indica que tan confiables fueron estas estimaciones 
            </p>''',unsafe_allow_html=True)

    st.write(f'''
    ## **Supuestos de media de los residuos y homoseasticidad:**
    ''',unsafe_allow_html=True)

    col1, col2 = st.columns([0.7,0.3])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred.flatten(), y=residuos.values.flatten(), mode="markers", name="Residuos",marker=dict(color="lightblue", size=7)))
        fig.add_trace(go.Scatter(x=y_pred.flatten(), y=np.repeat(np.mean(residuos), len(y_pred)), mode="lines", name="Media de Residuos", line=dict(color="red", dash="dash")))
        fig.update_layout(title="Residuos vs Valores Predichos", xaxis_title="Valores Predichos", yaxis_title="Residuos",
                          legend=dict(
                                        orientation="h",
                                        yanchor="top",
                                        y=-0.2,
                                        xanchor="center",
                                        x=0.5
                                    ),
                            height = 600)
        st.plotly_chart(fig)

    with col2:
            
        exog = sm.add_constant(X_train)
        bp_test = het_breuschpagan(residuos, exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        bp_results = dict(zip(labels, bp_test))
        
        bp_results_df = pd.DataFrame(list(bp_results.items()), columns=["Prueba", "Valor"])

                # Extraer solo el estadístico LM y su p-valor
        bp_statistic = bp_results['LM Statistic']
        bp_p_value = bp_results['LM-Test p-value']

        # Crear un DataFrame solo con el estadístico LM y su p-valor
        bp_table_df = pd.DataFrame({
            "Estadístico": ["LM Statistic"],
            "Valor": [bp_statistic],
            "P-valor": [bp_p_value]
        })

        # Aplicar el color según el p-valor
        styled_bp_table = bp_table_df.style.applymap(
            lambda val: 'background-color: #B0ED8B' if isinstance(val, (int, float)) and val < 0.05 else 
                        'background-color: #ED8181' if isinstance(val, (int, float)) else '',
            subset=["P-valor"]
        )

        st.table(styled_bp_table)

        st.write(f'''
        <p style='font-size:20px;'><b>media de los residuos:</b> la media de los residuos es {np.mean(residuos)} por lo que este supuesto se cumple<br>
        <b>homoseasticidad:</b> Para evaluar este supuesto se realiza la grafica de los residuos vs los valores predichos donde por la naturaleza
        de la variable objetivo se ven lineas extrañas pero de lo que se puede interpretar es que no tienen una varianza constante, ademas
        se evalua mediante la prueba breushpagan donde Un p-valor por debajo de 0.05 indica presencia de heterocedasticidad, que es este modelo
        es el caso 
        </p>''',unsafe_allow_html=True)


    st.write(f'''
    ## **Supuestos de Independencia de los residuos:**
    ''',unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.7,0.3])

    with col1:

        observaciones = np.arange(len(residuos))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=observaciones, y=residuos.values.flatten(), mode="lines+markers", name="Residuos",marker=dict(color="lightblue")))
        fig.add_trace(go.Scatter(x=observaciones, y=np.zeros(len(observaciones)), mode="lines", name="Referencia", line=dict(color="red", dash="dash")))
        fig.update_layout(title="Residuos vs Observaciones", xaxis_title="Observaciones", yaxis_title="Residuos")
        st.plotly_chart(fig)

    with col2:
        
        # Estadístico de Durbin-Watson
        dw_stat = durbin_watson(residuos)

        # Crear el DataFrame con el resultado
        dw_table_df = pd.DataFrame({
            "Estadístico": ["Durbin-Watson"],
            "Valor": [dw_stat[0]],
        })

        # Aplicar el color según el valor de Durbin-Watson
        def colorize_dw(val):
            if isinstance(val, (int, float)):
                if abs(val - 2) < 0.5:  # Consideramos independencia si DW está cerca de 2
                    return 'background-color: #B0ED8B'  # Verde (independientes)
                else:
                    return 'background-color: #ED8181'  # Rojo (dependientes)
            return ''

        # Aplicar el estilo
        styled_dw_table = dw_table_df.style.applymap(colorize_dw, subset=["Valor"])

        # Mostrar la tabla estilizada en Streamlit
        _, col1, _ = st.columns([0.2, 1, 0.2])
        st.table(styled_dw_table)

        st.write(f'''
        <p style='font-size:20px;'> Para evaluar la autocorrelacion de los residuos se hace la grafica ordenada de las predicciones vs
        los residuos y ademas se usa la prueba durbing watson para evaluarlos estadisticamente, dando como resultado tanto visual como 
        estadisticamente que no existe una autocorrelacion en los residuos
        </p>''',unsafe_allow_html=True)

    # Q-Q plot
    theoretical_quantiles = np.linspace(0.001, 0.999, len(residuos.values.flatten()))
    theoretical_values = stats.norm.ppf(theoretical_quantiles)
    residuals_sorted = np.sort(residuos.values.flatten())

    slope, intercept = np.polyfit(theoretical_values, residuals_sorted, 1)

    st.write("## **Supuesto de normalidad en los residuos**")


    col1, col2 = st.columns([0.7,0.3])

    with col1:
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

    with col2:
        residuos_array = residuos.values.flatten()

        shapiro_p = stats.shapiro(residuos_array)[1]
        ks_p = stats.kstest(residuos_array, 'norm')[1]
        dagostino_p = stats.normaltest(residuos_array)[1]

        test_results = {
            'Prueba de normalidad': ['Shapiro-Wilk', 'Kolmogorov-Smirnov', "D'Agostino-Pearson", ],
            'p-valor': [shapiro_p, ks_p, dagostino_p, ]
        }

        df = pd.DataFrame(test_results)

        def color_pvalue(val):
            color = '#B0ED8B' if val > 0.05 else '#ED8181'
            return f'background-color: {color}'

        st.table(df.style.applymap(color_pvalue, subset=['p-valor']))

        st.write(f'''
        <p style='font-size:20px;'>Otro supuesto es el comportamiento normal de los residuos, aunque visualmente se podria pensar
        que si cumple esta distribucion, las pruebas de normalidad indican lo contrario 
        </p>''',unsafe_allow_html=True)        

    st.write("## **Supuesto de ausencia de multicolinealidad**")

    # Factor de Inflación de la Varianza (VIF)
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X_train.columns
    vif_data['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    def highlight_vif(val):
        color = '#ED8181' if val > 10 else '#B0ED8B'
        return f'background-color: {color}'

    # Aplicar el estilo solo en la columna 'VIF' y mostrar la tabla en Streamlit
    styled_vif_data = vif_data.style.applymap(highlight_vif, subset=["VIF"])

    col1, col2 = st.columns([0.7,0.3])

    with col1:
        st.table(styled_vif_data)

    with col2:

        if procesamiento == "Ninguno":
            
            st.write(f'''
            <p style='font-size:20px;'> Como se observa ahi variables con una multicolinealidad demasiado alta, por lo que existen
            otras variables que explican estas variables, por lo que se deberia tratar esto para ajustar de mejor manera el modelo 
            </p>''',unsafe_allow_html=True)        

        if procesamiento == "EDA":
            
            st.write(f'''
            <p style='font-size:20px;'> Como se puede ver, quitar las variables mencionadas al principio soluciono el problema de multicolinealidad 
            </p>''',unsafe_allow_html=True)



    ###### Regresión polinómica
    st.markdown("# Regresión Polinómica")

    st.write(f'''
    <p style='font-size:20px;'> Mediante la busqueda en grilla y k-fold se busca la mejor combinacion de potencias para ajustar
             el modelo de regresion logistica 
    </p>''',unsafe_allow_html=True)

    st.code('''
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
            
    print("Mejor grado de polinomio encontrado:", best_degree)
            ''')

    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('linear_regression', LinearRegression())
    ])
    param_grid = {'poly_features__degree': [2]}
    kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kfolds, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_degree = grid_search.best_params_['poly_features__degree']
    best_model = grid_search.best_estimator_
    st.write(f"### Mejor grado de polinomio encontrado: {best_degree}")

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

    st.write("## **Coeficientes estimados**")

    _,col1,_ = st.columns([0.2,1,0.2])

    with col1:
        st.table(styled_coef_table)

    y_pred_poly = ols_poly_results.predict(X_train_poly_const)
    rmse_poly = np.sqrt(mean_squared_error(y_train, y_pred_poly))
    r2_poly = r2_score(y_train, y_pred_poly)
    
    y_pred_poly = best_model.predict(X_train)
    rmse_poly = np.sqrt(mean_squared_error(y_train, y_pred_poly))
    r2_poly = r2_score(y_train, y_pred_poly)


    # Gráfico de residuos vs valores predichos para regresión polinómica
    residuos_poly = y_train - y_pred_poly

    st.write(f'''
    ## **Supuestos de media de los residuos y homoseasticidad:**
    ''',unsafe_allow_html=True)

    col1, col2 = st.columns([0.7,0.3])

    with col1:

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred_poly.flatten(), y=residuos_poly.values.flatten(), mode="markers", name="Residuos",marker=dict(color='lightblue',size = 8)))
        fig.add_trace(go.Scatter(x=y_pred_poly.flatten(), y=np.repeat(np.mean(residuos_poly), len(y_pred_poly)), mode="lines", name="Media de Residuos", line=dict(color="red", dash="dash")))
        fig.update_layout(title="Residuos vs Valores Predichos (Regresión Polinómica)", xaxis_title="Valores Predichos", yaxis_title="Residuos",height = 600)
        st.plotly_chart(fig)

    with col2:
            
        exog = sm.add_constant(X_train)
        bp_test = het_breuschpagan(residuos_poly, exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        bp_results = dict(zip(labels, bp_test))
        
        bp_results_df = pd.DataFrame(list(bp_results.items()), columns=["Prueba", "Valor"])

                # Extraer solo el estadístico LM y su p-valor
        bp_statistic = bp_results['LM Statistic']
        bp_p_value = bp_results['LM-Test p-value']

        # Crear un DataFrame solo con el estadístico LM y su p-valor
        bp_table_df = pd.DataFrame({
            "Estadístico": ["LM Statistic"],
            "Valor": [bp_statistic],
            "P-valor": [bp_p_value]
        })

        # Aplicar el color según el p-valor
        styled_bp_table = bp_table_df.style.applymap(
            lambda val: 'background-color: #B0ED8B' if isinstance(val, (int, float)) and val < 0.05 else 
                        'background-color: #ED8181' if isinstance(val, (int, float)) else '',
            subset=["P-valor"]
        )

        st.table(styled_bp_table)

        st.write(f'''
        <p style='font-size:20px;'><b>media de los residuos:</b> la media de los residuos es {np.mean(residuos_poly)} por lo que este supuesto se cumple<br>
        <b>homoseasticidad:</b> Para evaluar este supuesto se realiza la grafica de los residuos vs los valores predichos donde por la naturaleza
        de la variable objetivo se ven lineas extrañas pero de lo que se puede interpretar es que no tienen una varianza constante, ademas
        se evalua mediante la prueba breushpagan donde Un p-valor por debajo de 0.05 indica presencia de heterocedasticidad, que es este modelo
        es el caso 
        </p>''',unsafe_allow_html=True)

    st.write(f'''
    ## **Supuestos de Independencia de los residuos**
    ''',unsafe_allow_html=True)

    col1, col2 = st.columns([0.7,0.3])

    with col1:

        observaciones = np.arange(len(residuos_poly))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=observaciones, y=residuos_poly.values.flatten(), mode="lines+markers", name="Residuos",marker=dict(color="lightblue")))
        fig.add_trace(go.Scatter(x=observaciones, y=np.zeros(len(observaciones)), mode="lines", name="Referencia", line=dict(color="red", dash="dash")))
        fig.update_layout(title="Residuos vs Observaciones", xaxis_title="Observaciones", yaxis_title="Residuos")
        st.plotly_chart(fig)

    with col2:
        
        # Estadístico de Durbin-Watson
        dw_stat = durbin_watson(residuos_poly)

        # Crear el DataFrame con el resultado
        dw_table_df = pd.DataFrame({
            "Estadístico": ["Durbin-Watson"],
            "Valor": [dw_stat[0]],
        })

        # Aplicar el color según el valor de Durbin-Watson
        def colorize_dw(val):
            if isinstance(val, (int, float)):
                if abs(val - 2) < 0.5:  # Consideramos independencia si DW está cerca de 2
                    return 'background-color: #B0ED8B'  # Verde (independientes)
                else:
                    return 'background-color: #ED8181'  # Rojo (dependientes)
            return ''

        # Aplicar el estilo
        styled_dw_table = dw_table_df.style.applymap(colorize_dw, subset=["Valor"])

        # Mostrar la tabla estilizada en Streamlit
        _, col1, _ = st.columns([0.2, 1, 0.2])
        st.table(styled_dw_table)

        st.write(f'''
        <p style='font-size:20px;'> Para evaluar la autocorrelacion de los residuos se hace la grafica ordenada de las predicciones vs
        los residuos y ademas se usa la prueba durbing watson para evaluarlos estadisticamente, dando como resultado tanto visual como 
        estadisticamente que no existe una autocorrelacion en los residuos
        </p>''',unsafe_allow_html=True)

    theoretical_quantiles = np.linspace(0.001, 0.999, len(residuos_poly.values.flatten()))
    theoretical_values = stats.norm.ppf(theoretical_quantiles)
    residuals_sorted = np.sort(residuos_poly.values.flatten())

    slope, intercept = np.polyfit(theoretical_values, residuals_sorted, 1)

    st.write("## **Supuesto de normalidad en los residuos**")


    col1, col2 = st.columns([0.7,0.3])

    with col1:
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

    with col2:
        residuos_array = residuos_poly.values.flatten()

        shapiro_p = stats.shapiro(residuos_array)[1]
        ks_p = stats.kstest(residuos_array, 'norm')[1]
        dagostino_p = stats.normaltest(residuos_array)[1]

        test_results = {
            'Prueba de normalidad': ['Shapiro-Wilk', 'Kolmogorov-Smirnov', "D'Agostino-Pearson", ],
            'p-valor': [shapiro_p, ks_p, dagostino_p, ]
        }

        df = pd.DataFrame(test_results)

        def color_pvalue(val):
            color = '#B0ED8B' if val > 0.05 else '#ED8181'
            return f'background-color: {color}'

        st.table(df.style.applymap(color_pvalue, subset=['p-valor']))

        st.write(f'''
        <p style='font-size:20px;'>Otro supuesto es el comportamiento normal de los residuos, visualmente se podria pensar
        que si cumple esta distribucion esta vez las pruebas de normalidad si podriamos decir que con un poco mas de seguridad
        los residuos pueden seguir una distribucion normal 
        </p>''',unsafe_allow_html=True)        

    st.write("## **Supuesto de ausencia de multicolinealidad**")

    poly_columns = poly.get_feature_names_out(input_features=X_train.columns)
    X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_columns)

    # Calcular el Factor de Inflación de la Varianza (VIF)
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X_train_poly_df.columns
    vif_data['VIF'] = [variance_inflation_factor(X_train_poly_df.values, i) for i in range(X_train_poly_df.shape[1])]

    # Función para aplicar el estilo
    def highlight_vif(val):
        color = '#ED8181' if val > 10 else '#B0ED8B'
        return f'background-color: {color}'

    # Aplicar el estilo a la columna 'VIF'
    styled_vif_data = vif_data.style.applymap(highlight_vif, subset=["VIF"])

    # Mostrar la tabla en Streamlit
    col1, col2 = st.columns([0.7, 0.3])

    with col1:
        st.table(styled_vif_data)

    if procesamiento == "Ninguno":
        with col2:
                st.write(f'''
                <p style='font-size:20px;'> Como se menciono anteriormente en la seccion de multicolinealidad, en estos modelos es muy
                        comun que al tener tantos exponentes de una misma variables o combinacion entre varias, todos las variables 
                        estas muuuuy correlacionadas con las demas y esto es problematico como se hablo anteriormente y como se observo
                        en las estimaciones de los coeficientes de este modelo 
                </p>''',unsafe_allow_html=True)     
    else:
        with col2:
                st.write(f'''
                <p style='font-size:20px;'> Podemos ver que gracias al preprocesamiento, la multicolinealidad se elimino de una regresion
                        polinomica, que como lo discutimos en la pagina de multicolinealidad, quitar este efecto es muy importante para 
                        conseguir estimadores mas confiables  
                </p>''',unsafe_allow_html=True)


    st.write("# **Rendimientos**:")

    st.write("#### **Entrenamiento**:")
    y_pred = linear_model.predict(X_train)
    rmse_linear = np.sqrt(mean_squared_error(y_train, y_pred))
    r2_linear = r2_score(y_train, y_pred) #modelo base la media de y
    
    y_pred_poly = ols_poly_results.predict(X_train_poly_const)
    rmse_poly = np.sqrt(mean_squared_error(y_train, y_pred_poly))
    r2_poly = r2_score(y_train, y_pred_poly)

    # Crear el DataFrame con los resultados
    results = pd.DataFrame({
        'Modelo': ['Lineal', 'Polinómico'],
        'RMSE': [rmse_linear, rmse_poly],
        'R²': [r2_linear, r2_poly]
    })

    st.table(results)

    st.write("#### **validacion**:")
    y_pred = linear_model.predict(X_test)
    rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_linear = r2_score(y_test, y_pred) #modelo base la media de y
    
    X_test_poly = poly.fit_transform(X_test)
    X_test_poly_const = sm.add_constant(X_test_poly)

    y_pred_poly = ols_poly_results.predict(X_test_poly_const)
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    r2_poly = r2_score(y_test, y_pred_poly)

    # Crear el DataFrame con los resultados
    results = pd.DataFrame({
        'Modelo': ['Lineal', 'Polinómico'],
        'RMSE': [rmse_linear, rmse_poly],
        'R²': [r2_linear, r2_poly]
    })

    st.table(results)

    st.write(f'''
            <p style='font-size:20px;'> Se observa que en validacion el modelo de regresion lineal puedo generalizar de mejor manera que
             el polinomico, esto podria ser por que el polinomico tiene una facilidad para sobreajustarse  
        </p>''',unsafe_allow_html=True)

    st.write("# **Conclusiones**:")

    st.write(f'''
            <p style='font-size:20px;'> Se realizo el ejercicio para aplicar lo aprendido, pero la naturaleza de la variable objetivo 
            hacia que no tuviera sentido aplicar una regresion de estos tipos, pues la variable se asemeja mas a una variable categorica
            que a una continua que serviria mas para estos modelos 
        </p>''',unsafe_allow_html=True)

    st.write(f'''
            <p style='font-size:20px;'> Como se pudo ver en los analisis y en los resultados de los modelos, la regresion lineal fue 
            mejor que la logistica por que aunque se inclute potencias y una capacidad de adptarse a datos no lineales, los coeficientes
            y los rendimientos dan como ganador a la regresion lineal simple 
        </p>''',unsafe_allow_html=True)
    
elif problema == "Clasificacion":

    st.write("# Regresion Logistica")

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

    modelo = st.selectbox("Selecciona modelo:",
                        ["Regresion logistica","KNN","Comparacion"])

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
    
    if modelo == "Regresion logistica":

        X_scaled = sm.add_constant(X_scaled)

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
            fig_train = go.Figure(data=go.Heatmap(
                z=conf_matrix_train,
                x=["Predicho 0", "Predicho 1"],
                y=["Real 0", "Real 1"],
                colorscale="Blues",
                hoverongaps=False,
                text=conf_matrix_train,  # Agregar los valores en cada celda
                texttemplate="%{text}",   # Formato de texto
                showscale=True
            ))
            fig_train.update_layout(title="Matriz de Confusión (Entrenamiento)", xaxis_title="Predicción", yaxis_title="Real")
            st.plotly_chart(fig_train)

        with col2:
            fig_test = go.Figure(data=go.Heatmap(
                z=conf_matrix_test,
                x=["Predicho 0", "Predicho 1"],
                y=["Real 0", "Real 1"],
                colorscale="Blues",
                hoverongaps=False,
                text=conf_matrix_test,  # Agregar los valores en cada celda
                texttemplate="%{text}",  # Formato de texto
                showscale=True
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
    
    elif modelo == "KNN":
    
        st.write('''<p style='font-size:23px;'>
        Aplicamos la busqueda en grilla para buscar los mejores hiperparametros para el modelo de KNN 
        </p>''',unsafe_allow_html=True)

        st.code('''
        knn = KNeighborsClassifier()

        param_grid = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance'], 'p': [1, 2]}  #Manhattan,  Euclidiana
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
        grid_search.fit(X_scaled, y_train)

        best_knn = grid_search.best_estimator_
        print("Mejores parámetros:", grid_search.best_params_)
        ''')

        knn = KNeighborsClassifier()

        if balance == "Original":
            param_grid = {'n_neighbors': [11], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        elif balance == "ADASYN":
            param_grid = {'n_neighbors': [2], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        elif balance == "BORDERLINE_SMOTE":
            param_grid = {'n_neighbors': [2], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        elif balance == "RANDOM OVER SAMPLING":
            param_grid = {'n_neighbors': [1], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        elif balance == "RANDOM UNDER SAMPLING":
            param_grid = {'n_neighbors': [11], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        best_knn = grid_search.best_estimator_
        st.write("Mejores parámetros:", grid_search.best_params_)

        # Predicciones de probabilidad para el conjunto de entrenamiento y prueba
        y_pred_train_proba = best_knn.predict(X_scaled)
        y_pred_test_proba = best_knn.predict(scaler.transform(X_test))

        threshold = st.slider("Selecciona el umbral de predicción:", 0.0, 1.0, 0.5, 0.01)

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
                hoverongaps=False,
                text=conf_matrix_train,  # Agregar los valores en cada celda
                texttemplate="%{text}",   # Formato de texto
                showscale=True
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
                hoverongaps=False,
                text=conf_matrix_test,  # Agregar los valores en cada celda
                texttemplate="%{text}",  # Formato de texto
                showscale=True
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
    
    elif modelo == "Comparacion":  
        knn = KNeighborsClassifier()

        if balance == "Original":
            param_grid = {'n_neighbors': [11], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        elif balance == "ADASYN":
            param_grid = {'n_neighbors': [2], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        elif balance == "BORDERLINE_SMOTE":
            param_grid = {'n_neighbors': [2], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        elif balance == "RANDOM OVER SAMPLING":
            param_grid = {'n_neighbors': [1], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        elif balance == "RANDOM UNDER SAMPLING":
            param_grid = {'n_neighbors': [11], 'weights': ['uniform', 'distance'], 'p': [1]}  #Manhattan,  Euclidiana
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=kf, scoring='accuracy')
            grid_search.fit(X_scaled, y_train)

        best_knn = grid_search.best_estimator_

        y_pred_train_proba = best_knn.predict(X_scaled)
        y_pred_test_proba = best_knn.predict(scaler.transform(X_test))

        threshold = st.slider("Selecciona el umbral de predicción:", 0.0, 1.0, 0.5, 0.01)

        y_pred_train = (y_pred_train_proba >= threshold).astype(int)
        y_pred_test = (y_pred_test_proba >= threshold).astype(int)

        datos = st.selectbox("Seleccione el conjunto de datos: ", ["Entrenamiento", "Validacion"])

        col1, col2 = st.columns(2)

        if datos == "Entrenamiento":
            
            report = classification_report(y_train, y_pred_train, output_dict=True)

            report_df = pd.DataFrame(report).transpose()

            with col1:
                st.write("### Classification Report KNN")
                st.table(report_df)

        else:

            report = classification_report(y_test, y_pred_test, output_dict=True)

            report_df = pd.DataFrame(report).transpose()

            with col1:
                st.write("### Classification Report KNN")
                st.table(report_df)

        X_scaled = sm.add_constant(X_scaled)

        log_reg_sm = sm.Logit(y_train, X_scaled).fit()
        y_pred_train_proba = log_reg_sm.predict(X_scaled)
        y_pred_test_proba = log_reg_sm.predict(sm.add_constant(scaler.transform(X_test), has_constant='add'))

        # Convertir probabilidades en etiquetas según el umbral seleccionado
        y_pred_train = (y_pred_train_proba >= threshold).astype(int)
        y_pred_test = (y_pred_test_proba >= threshold).astype(int)

        if datos == "Entrenamiento":

            report = classification_report(y_train, y_pred_train, output_dict=True)

            report_df = pd.DataFrame(report).transpose()

            with col2:
                st.write("### Classification Report regresion logistica")
                st.table(report_df)

        else:

            report = classification_report(y_test, y_pred_test, output_dict=True)

            report_df = pd.DataFrame(report).transpose()

            with col2:
                st.write("### Classification Report regresion logistica")
                st.table(report_df)

        st.write("# **Conclusiones**")

        st.write('''<p style='font-size:23px;'>
        En general, en entrenamiento pareciera que el modelo de KNN se sobreajusto a los datos con los parametros encontrados en la busqueda en grilla
        pero aun asi su rendimiento en valdicion es mejor que el del modelo logistico, aunque se puede evidenciar que cuando se balancean los datos, el
        modelo logistico logra mejores resultados, cosa que el KNN no necesito para funcionar bien, por lo que para estos datos se adapta mejor un KNN para
        la clasificacion de los vinos
        </p>''',unsafe_allow_html=True)
