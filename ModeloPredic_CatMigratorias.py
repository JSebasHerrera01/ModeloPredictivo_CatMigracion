import streamlit as st
from st_files_connection import FilesConnection
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import boto3
from io import BytesIO
from matplotlib.ticker import MaxNLocator

# Centrar el título con Markdown y Streamlit
st.markdown("<h1 style='text-align: center;'>Análisis Predictivo de Categorias Migratorias en Colombia</h1>", unsafe_allow_html=True)

# Crear conexión a S3
conn = st.connection('s3', type=FilesConnection)
# Descargar archivo CSV desde S3
turismo_data = conn.read("streamlitbuckett/data.csv", input_format="csv", ttl=600, encoding='utf-8', lineterminator='\n')







# Filtrar los datos para  "Entradas"
turismo_df = turismo_df[(turismo_df['Entrada'] == 'Entradas')]

# Crear y entrenar un modelo para cada región
regiones = turismo_df['Region_Nacionalidad'].unique()

# Crear y entrenar un modelo para cada categoría migratoria
categorias_migratorias = turismo_df['Categoría_Migratoria'].unique()

# Añadir un cuadro de texto en la barra lateral
st.sidebar.title("Motivos de Viaje por cada Categoría Migratoria")

# Agrupar por categoría migratoria y motivo de viaje
motivos_viaje_por_categoria = turismo_df.groupby(['Categoría_Migratoria', 'Motivo_Viaje'])['Total_Personas'].sum().reset_index()

# Iterar sobre cada categoría migratoria
for categoria in categorias_migratorias:
    st.sidebar.write(f"**{categoria}**")
    
    # Filtrar los datos para la categoría migratoria actual
    motivos_viaje_categoria = motivos_viaje_por_categoria[motivos_viaje_por_categoria['Categoría_Migratoria'] == categoria]
    
    # Obtener el top 5 de motivos de viaje por cantidad de personas
    top5_motivos_viaje = motivos_viaje_categoria.nlargest(5, 'Total_Personas')
    
    # Concatenar los motivos de viaje separados por coma
    motivos_viaje_texto = ", ".join(top5_motivos_viaje['Motivo_Viaje'])
    
    # Mostrar los top 5 motivos de viaje para la categoría migratoria actual
    st.sidebar.write(motivos_viaje_texto)

# Almacenar predicciones y métricas para cada región
predicciones_regionales = {}
metricas_regionales = {}

# Menú desplegable para seleccionar el modelo
modelo_seleccionado = st.selectbox('Selecciona el modelo de análisis', ('Regresión Lineal', 'Random Forest', 'SVM', 'XGBoost'))

# Menú desplegable para seleccionar la región
# Menú desplegable para seleccionar la región o categoría migratoria
seleccion = st.selectbox('Selecciona una categoría migratoria:', ['Todas las categorias'] + list(categorias_migratorias))

if seleccion == 'Todas las categorias':
        # Filtrar datos para todas las categorías y todas las regiones
        total_personas_por_año_todas_categorias = turismo_df.groupby(['Año', 'Region_Nacionalidad'])['Total_Personas'].sum().reset_index()

        # Calcular el total de personas para todas las regiones y categorías migratorias
        total_personas_por_año_todas_categorias_total = total_personas_por_año_todas_categorias.groupby('Año')['Total_Personas'].sum().reset_index()

        # Entrenar el modelo seleccionado para todas las categorías y todas las regiones
        if modelo_seleccionado == 'Regresión Lineal':
            model = LinearRegression()
        elif modelo_seleccionado == 'Random Forest':
            model = RandomForestRegressor()
        elif modelo_seleccionado == 'SVM':
            model = SVR()
        elif modelo_seleccionado == 'XGBoost':
            model = xgb.XGBRegressor()

        model.fit(total_personas_por_año_todas_categorias_total[['Año']], total_personas_por_año_todas_categorias_total['Total_Personas'])

        # Realizar predicciones para el año 2024
        prediction_2024_todas_categorias_total = model.predict([[2024]])

        # Gráfico de barras para todas las categorías y todas las regiones
        plt.figure(figsize=(10, 6))
        plt.bar(total_personas_por_año_todas_categorias_total['Año'], total_personas_por_año_todas_categorias_total['Total_Personas'], color='gray', label='Historico - Todas las Categorías y Regiones')
        plt.bar(2024, prediction_2024_todas_categorias_total[0], color='green', label='Predicción 2024 - Todas las Categorías y Regiones', alpha=0.7)
        plt.xlabel('Año')
        plt.ylabel('Total de Personas')
        plt.title(f'Tendencias Históricas y Predicción para Todas las Categorías (Turismo - Entradas) - Todas las Regiones')
        plt.legend()
        # Forzar que los ticks del eje x sean enteros
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Métricas de rendimiento
        y_pred_todas_categorias_total = model.predict(total_personas_por_año_todas_categorias_total[['Año']])
        mse_todas_categorias_total = mean_squared_error(total_personas_por_año_todas_categorias_total['Total_Personas'], y_pred_todas_categorias_total)
        r2_todas_categorias_total = r2_score(total_personas_por_año_todas_categorias_total['Total_Personas'], y_pred_todas_categorias_total)

        # Mostrar la predicción y las métricas para todas las categorías y todas las regiones
        st.markdown(f"**Predicción para el año 2024 (Todas las Categorías) - Todas las Regiones ({modelo_seleccionado}):**")
        st.markdown(f"**{prediction_2024_todas_categorias_total[0]:,.0f}** personas")
        st.markdown(f"**Métricas de Efectividad (Todas las Categorías) - Todas las Regiones ({modelo_seleccionado}):**")
        st.markdown(f"MSE: {mse_todas_categorias_total:,.2f}")
        st.markdown(f"RMSE: {np.sqrt(mse_todas_categorias_total):,.2f}")
        st.markdown(f"R²: {r2_todas_categorias_total:,.2f}")

        # Mostrar el gráfico
        st.pyplot(plt)


elif seleccion in categorias_migratorias:
            # Filtrar datos solo para la categoría migratoria seleccionada
    categoria_df = turismo_df[turismo_df['Categoría_Migratoria'] == seleccion]
    total_personas_por_año_categoria = categoria_df.groupby(['Año', 'Region_Nacionalidad'])['Total_Personas'].sum().reset_index()

    # Mostrar el selectbox para filtrar por región
    region_seleccionada_categoria = st.selectbox('Selecciona una región:', ['Todas las regiones'] + list(regiones))

    if region_seleccionada_categoria == 'Todas las regiones':
        # Calcular el total de personas para todas las regiones y la categoría migratoria seleccionada
        total_personas_por_año_categoria_total = total_personas_por_año_categoria.groupby('Año')['Total_Personas'].sum().reset_index()

        # Entrenar el modelo seleccionado para la categoría migratoria seleccionada y todas las regiones
        if modelo_seleccionado == 'Regresión Lineal':
            model = LinearRegression()
        elif modelo_seleccionado == 'Random Forest':
            model = RandomForestRegressor()
        elif modelo_seleccionado == 'SVM':
            model = SVR()
        elif modelo_seleccionado == 'XGBoost':
            model = xgb.XGBRegressor()

        model.fit(total_personas_por_año_categoria_total[['Año']], total_personas_por_año_categoria_total['Total_Personas'])

        # Realizar predicciones para el año 2024
        prediction_2024_categoria_total = model.predict([[2024]])

        # Gráfico de barras para la categoría migratoria seleccionada y todas las regiones
        plt.figure(figsize=(10, 6))
        plt.bar(total_personas_por_año_categoria_total['Año'], total_personas_por_año_categoria_total['Total_Personas'], color='gray', label=f'{seleccion} - Historico - Todas las Regiones')
        plt.bar(2024, prediction_2024_categoria_total[0], color='green', label=f'{seleccion} - Predicción 2024 - Todas las Regiones', alpha=0.7)
        plt.xlabel('Año')
        plt.ylabel('Total de Personas')
        plt.title(f'Tendencias Históricas y Predicción para {seleccion} (Turismo - Entradas) - Todas las Regiones')
        plt.legend()
        # Forzar que los ticks del eje x sean enteros
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))



        # Métricas de rendimiento
        y_pred_categoria_total = model.predict(total_personas_por_año_categoria_total[['Año']])
        mse_categoria_total = mean_squared_error(total_personas_por_año_categoria_total['Total_Personas'], y_pred_categoria_total)
        r2_categoria_total = r2_score(total_personas_por_año_categoria_total['Total_Personas'], y_pred_categoria_total)

        # Mostrar la predicción y las métricas para la categoría migratoria y todas las regiones
        st.markdown(f"**Predicción para el año 2024 en {seleccion} ({modelo_seleccionado}) - Todas las Regiones:**")
        st.markdown(f"**{prediction_2024_categoria_total[0]:,.0f}** personas")
        st.markdown(f"**Métricas de Efectividad para {seleccion} ({modelo_seleccionado}) - Todas las Regiones:**")
        st.markdown(f"MSE: {mse_categoria_total:,.2f}")
        st.markdown(f"RMSE: {np.sqrt(mse_categoria_total):,.2f}")
        st.markdown(f"R²: {r2_categoria_total:,.2f}")

        # Mostrar el gráfico
        st.pyplot(plt)

    elif region_seleccionada_categoria in regiones:
        # Filtrar datos para la categoría migratoria y la región seleccionada
        categoria_region_df = categoria_df[categoria_df['Region_Nacionalidad'] == region_seleccionada_categoria]
        total_personas_por_año_categoria_region = categoria_region_df.groupby('Año')['Total_Personas'].sum().reset_index()

        # Entrenar el modelo seleccionado para la categoría migratoria y la región seleccionada
        if modelo_seleccionado == 'Regresión Lineal':
            model = LinearRegression()
        elif modelo_seleccionado == 'Random Forest':
            model = RandomForestRegressor()
        elif modelo_seleccionado == 'SVM':
            model = SVR()
        elif modelo_seleccionado == 'XGBoost':
            model = xgb.XGBRegressor()

        model.fit(total_personas_por_año_categoria_region[['Año']], total_personas_por_año_categoria_region['Total_Personas'])

        # Realizar predicciones para el año 2024
        prediction_2024_categoria_region = model.predict([[2024]])

        # Gráfico de barras para la categoría migratoria y la región seleccionada
        plt.figure(figsize=(10, 6))
        plt.bar(total_personas_por_año_categoria_region['Año'].astype(int), total_personas_por_año_categoria_region['Total_Personas'], color='gray', label=f'{seleccion} - Historico - {region_seleccionada_categoria}')
        plt.bar(2024, prediction_2024_categoria_region[0], color='green', label=f'{seleccion} - Predicción 2024 - {region_seleccionada_categoria}', alpha=0.7)
        plt.xlabel('Año')
        plt.ylabel('Total de Personas')
        plt.title(f'Tendencias Históricas y Predicción para {seleccion} (Turismo - Entradas) - {region_seleccionada_categoria}')
        plt.legend()
        # Forzar que los ticks del eje x sean enteros
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Métricas de rendimiento
        y_pred_categoria_region = model.predict(total_personas_por_año_categoria_region[['Año']])
        mse_categoria_region = mean_squared_error(total_personas_por_año_categoria_region['Total_Personas'], y_pred_categoria_region)
        r2_categoria_region = r2_score(total_personas_por_año_categoria_region['Total_Personas'], y_pred_categoria_region)

        # Mostrar la predicción y las métricas para la categoría migratoria y la región seleccionada
        st.markdown(f"**Predicción para el año 2024 en {seleccion} ({modelo_seleccionado}) - {region_seleccionada_categoria}:**")
        st.markdown(f"**{prediction_2024_categoria_region[0]:,.0f}** personas")
        st.markdown(f"**Métricas de Efectividad para {seleccion} ({modelo_seleccionado}) - {region_seleccionada_categoria}:**")
        st.markdown(f"MSE: {mse_categoria_region:,.2f}")
        st.markdown(f"RMSE: {np.sqrt(mse_categoria_region):,.2f}")
        st.markdown(f"R²: {r2_categoria_region:,.2f}")

        # Mostrar el gráfico
        st.pyplot(plt)
