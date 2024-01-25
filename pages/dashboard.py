import streamlit as st

# Código Streamlit
st.title("Visualización Power BI")

# Código HTML para el iframe
iframe_code = """
<iframe title="AnalisisExtranjeros_Entradas_24/01" width="1140" height="541.25" 
src="https://app.powerbi.com/reportEmbed?reportId=87fc7844-5a79-4ee7-acf7-8ed15a2938b8&autoAuth=true&ctid=bce86336-c485-448d-94ef-1c2df68ef035" 
frameborder="0" allowFullScreen="true"></iframe>
"""

# Mostrar el iframe en la aplicación
st.markdown(iframe_code, unsafe_allow_html=True)
