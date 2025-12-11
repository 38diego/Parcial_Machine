# Parcial de Machine Learning - Aplicación Práctica de Regresión y Clasificación

Esta aplicación web interactiva presenta la solución al parcial de Machine Learning. El proyecto tiene un enfoque práctico y visual para entender y comparar los fundamentos de los modelos de regresión y clasificación utilizando **Python** y **R**.

## Objetivos del Proyecto

La aplicación permite explorar conceptos clave de Machine Learning a través de 4 módulos interactivos:

### 1. Comparativa de Herramientas (Python vs R)
*   **Objetivo**: Entender las diferencias y similitudes al implementar Regresión Lineal en dos de los lenguajes más usados en ciencia de datos.
*   **Lo que verás**:
    *   Comparación visual entre métodos de ajuste (Mínimos Cuadrados vs Descenso del Gradiente).
    *   Animaciones que muestran cómo aprende el modelo paso a paso.
    *   Explicación práctica de los parámetros de configuración en librerías como `scikit-learn` y `glmnet`.

### 2. Impacto de la Multicolinealidad
*   **Objetivo**: Visualizar qué sucede cuando las variables de un modelo están demasiado correlacionadas y cómo solucionarlo.
*   **Lo que verás**:
    *   Simulaciones interactivas de regresión polinómica.
    *   Comparación entre el uso de polinomios estándar y polinomios de Chebyshev para mejorar la estabilidad del modelo.

### 3. Caso de Estudio: Calidad del Vino
*   **Objetivo**: Aplicar técnicas de regresión y clasificación en un conjunto de datos real (*WineQT*).
*   **Lo que verás**:
    *   **Regresión**: Cómo el preprocesamiento y el análisis exploratorio (EDA) mejoran las predicciones.
    *   **Clasificación**: Estrategias para manejar datos desbalanceados y comparar el rendimiento de modelos como Regresión Logística y KNN.

### 4. Laboratorio de Diagnóstico de Modelos
*   **Objetivo**: Aprender a validar si un modelo de regresión es confiable.
*   **Lo que verás**:
    *   Diagnóstico paso a paso sobre el dataset *Boston Housing*.
    *   Gráficos para verificar los supuestos estadísticos (normalidad de errores, varianza constante, etc.) y entender por qué son importantes.

## Ejecución

Para explorar la aplicación, ejecuta el siguiente comando en tu terminal:

```bash
streamlit run app.py
```

O ir al [streamlit cloud](https://parcialmachine.streamlit.app/)