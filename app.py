import streamlit as st

st.set_page_config(layout="wide")  # Debe ser la primera llamada

# Definir las páginas como opciones
paginas = {
    "Caso de estudio": "pages/caso_estudio.py",
    "Laboratorio supuestos": "pages/Laboratorio_supuestos.py",
    "Multicolinealidad": "pages/Multicolinealidad.py",
    "Regresión Lineal": "pages/Regresion_Lineal.py"
}

st.sidebar.title("Actividades")
seleccion = st.sidebar.selectbox("Seleccione una página", list(paginas.keys()))

# Ejecutar la página seleccionada
pagina_seleccionada = paginas[seleccion]
exec(open(pagina_seleccionada).read())