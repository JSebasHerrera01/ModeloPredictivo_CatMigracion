import streamlit as st

# Código Streamlit
st.markdown("<h1 style='text-align: center;'>Visualización de Flujos Migratorios en Colombia (Entradas al pais)</h1>", unsafe_allow_html=True)

# Mostrar el Iframe
iframe_code = """
<div style="display: flex; justify-content: center; align-items: center; height: 90vh; width: 65vw; overflow: hidden; margin: auto;">
    <iframe title="AnalisisExtranjeros_Entradas_24/01" style="width: 100%; height: 100%; margin: 0;" 
    src="https://app.powerbi.com/reportEmbed?reportId=87fc7844-5a79-4ee7-acf7-8ed15a2938b8&autoAuth=true&ctid=bce86336-c485-448d-94ef-1c2df68ef035" 
    frameborder="0" allowFullScreen="true"></iframe>
</div>
"""

# Mostrar el iframe centrado y ajustado en la aplicación
st.markdown(iframe_code, unsafe_allow_html=True)
