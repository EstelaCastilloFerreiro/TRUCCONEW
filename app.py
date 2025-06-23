import streamlit as st
st.set_page_config(page_title="TRUCCO", page_icon="🡕", layout="wide")

from dashboard import mostrar_dashboard
import pandas as pd
import base64
import os

# Function to get absolute path for assets
def get_asset_path(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "assets", filename)

# Estilos CSS
st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        padding: 20px 0;
        margin-bottom: 40px;
    }
    .logo-container {
        margin-right: 30px;
    }
    .main-title {
        font-size: 48px;
        color: #666666;
        font-weight: 600;
        line-height: 1;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin: 0;
        padding: 0;
    }
    .login-title {
        font-size: 26px;
        font-weight: 500;
        margin-bottom: 1rem;
        color: white;
    }
    .stApp {
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Función para aplicar fondo
def set_background(image_file):
    """Establece una imagen de fondo para la app y aplica estilos de login."""
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
            b64_encoded = base64.b64encode(img_data).decode()
            style = f"""
                <style>
                .stApp {{
                    background-image: url(data:image/png;base64,{b64_encoded});
                    background-size: cover;
                }}
                .login-container {{
                    background-color: rgba(255, 255, 255, 0.8);
                    padding: 30px;
                    border-radius: 10px;
                    text-align: center;
                }}
                .login-title {{
                    font-size: 24px;
                    font-weight: bold;
                    color: white;
                    margin-bottom: 20px;
                }}
                /* Poner etiquetas de input en blanco */
                div[data-testid="stTextInput"] label {{
                    color: white;
                }}
                </style>
            """
            st.markdown(style, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading background image: {e}")

# Login de seguridad
if 'logueado' not in st.session_state:
    try:
        st.markdown(f'''
            <div style="display: flex; align-items: center; width: 100%; margin-bottom: 40px;">
                <img src="data:image/png;base64,{base64.b64encode(open(get_asset_path('Logo.png'), 'rb').read()).decode()}" style="height: 160px; margin-right: 48px;" />
                <img src="data:image/png;base64,{base64.b64encode(open(get_asset_path('fondo.png'), 'rb').read()).decode()}" style="height: 240px; width: 100%; object-fit: cover;" />
            </div>
        ''', unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #666666; margin-bottom: 20px;'>Trucco Analytics</h1>", unsafe_allow_html=True)
        st.markdown("<div class='login-title'>Acceso a TRUCCO Analytics</div>", unsafe_allow_html=True)
        usuario = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        if st.button("Entrar"):
            if usuario and password:
                st.session_state['logueado'] = True
            else:
                st.warning("Por favor, introduce usuario y contraseña")
    except Exception as e:
        st.error(f"Error loading login assets: {e}")

else:
    # Ya logueado
    try:
        with open(get_asset_path("Logo.png"), "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
                <div class="header-container">
                    <div class="logo-container">
                        <img src="data:image/png;base64,{logo_data}" width="100">
                    </div>
                    <h1 class="main-title">Plataforma de Análisis y Predicción</h1>
                </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading header logo: {e}")

    st.sidebar.title("Menú de Navegación")
    opcion = st.sidebar.radio("Selecciona una vista", ["Análisis", "Predicción"])

    # Subida de archivo solo después del login
    file = st.sidebar.file_uploader("Sube el archivo Excel", type=["xlsx"])

    if file:
        try:
            xls = pd.ExcelFile(file, engine="openpyxl")
            df_productos = xls.parse("Compra")
            df_traspasos = xls.parse("Traspasos de almacén a tienda")
            df_ventas = xls.parse("ventas 23 24 25")

            st.sidebar.success("Archivo cargado correctamente")

            if opcion == "Análisis":
                seccion = st.sidebar.selectbox("Área de Análisis", [
                    "Resumen General",
                    "Geográfico y Tiendas",
                    "Stock y Traspasos",
                    "Producto, Campaña, Devoluciones y Rentabilidad"
                ])
                mostrar_dashboard(df_productos, df_traspasos, df_ventas, seccion)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("Sube el archivo Excel para comenzar.")
