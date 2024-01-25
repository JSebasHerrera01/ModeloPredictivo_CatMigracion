import streamlit as st

# Código Streamlit
st.title("Visualización de Flujos Migratorios en Colombia (Entradas al pais)")

# Mostrar el Iframe
iframe_code = """
<div style="display: flex; justify-content: center; align-items: center; height: 80vh; width: 80vw; overflow: hidden;">
    <iframe title="AnalisisExtranjeros_Entradas_24/01" style="width: 100%; height: 100%;" 
    src="https://app.powerbi.com/reportEmbed?reportId=87fc7844-5a79-4ee7-acf7-8ed15a2938b8&autoAuth=true&ctid=bce86336-c485-448d-94ef-1c2df68ef035" 
    frameborder="0" allowFullScreen="true"></iframe>
</div>
"""
# Mostrar el iframe en la aplicación
st.markdown(iframe_code, unsafe_allow_html=True)
