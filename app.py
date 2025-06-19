import streamlit as st
st.set_page_config(page_title="TRUCCO", page_icon="🡕", layout="wide")

from dashboard import mostrar_dashboard
import pandas as pd
import base64
import os

# Cargar imágenes al inicio
def load_image(path):
    try:
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image {path}: {e}")
        return None

# Cargar imágenes en session state si no están ya cargadas
if 'bg_image' not in st.session_state:
    st.session_state.bg_image = load_image("assets/bg_trucco.png")
if 'logo_image' not in st.session_state:
    st.session_state.logo_image = load_image("assets/trucco.png")

# Estilos CSS
st.markdown("""
    <style>
    body, .stApp, .main, .sidebar, .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2, .css-ffhzg2, .css-1offfwp, .css-1v3fvcr, .css-1lcbmhc, .css-1y4p8pa, .css-1n76uvr, .css-1b0udgb, .css-1q8dd3e, .css-1d391kg *, .stTextInput > div > input, .stTextInput > label, .stSelectbox > div, .stSelectbox > label, .stRadio > div, .stRadio > label, .stButton > button, .stFileUploader > div, .stFileUploader > label, .stAlert, .stAlert > div, .stAlert > span, .stSidebar, .stSidebar * {
        color: #000000 !important;
    }
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
        font-size: 120px;
        color: #000000;
        font-weight: 600;
        line-height: 1;
        letter-spacing: -2px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin: 0;
        padding: 0;
    }
    .login-title {
        font-size: 26px;
        font-weight: 500;
        margin-bottom: 1rem;
        color: #000000;
    }
    .stApp {
        background-color: white;
    }
    .stSidebar, section[data-testid="stSidebar"], .css-1d391kg, .stSidebarContent {
        background-color: white !important;
    }
    header, .st-emotion-cache-18ni7ap, .st-emotion-cache-1avcm0n {
        background-color: white !important;
    }
    .stFileUploader, .stFileUploader > div, .stFileUploader > label, .stFileUploader > span, .stFileUploader * {
        background-color: #fff !important;
        color: #000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Función para aplicar fondo
def set_background(encoded_image):
    if encoded_image:
        css = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            opacity: 0.96;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 1rem;
            border-radius: 10px;
        }}
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# Login de seguridad
if 'logueado' not in st.session_state:
    if st.session_state.bg_image:
        set_background(st.session_state.bg_image)  # Aplica fondo solo antes del login
    
    if st.session_state.logo_image:
        st.markdown(f'<img src="data:image/png;base64,{st.session_state.logo_image}" width="180">', unsafe_allow_html=True)
    
    # CSS para el login-title en blanco SOLO en login
    st.markdown("""
        <style>
        .login-title {
            font-size: 26px;
            font-weight: 500;
            margin-bottom: 1rem;
            color: #fff !important;
        }
        /* Hace las etiquetas de los inputs blancas solo en login */
        label, .stTextInput > label, .stTextInput label, .stTextInput div label {
            color: #fff !important;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='login-title'>Acceso a TRUCCO Analytics</div>", unsafe_allow_html=True)
    usuario = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        # Permitir acceso si usuario, contraseña o '0' en cualquiera
        if usuario == "0" or password == "0" or (usuario and password):
            st.session_state['logueado'] = True
        else:
            st.warning("Por favor, introduce usuario y contraseña")

else:
    # Ya logueado
    st.markdown("""
        <style>
        body, .stApp, .main, .sidebar, .css-1d391kg, .css-1v0mbdj, .css-1cpxqw2, .css-ffhzg2, .css-1offfwp, .css-1v3fvcr, .css-1lcbmhc, .css-1y4p8pa, .css-1n76uvr, .css-1b0udgb, .css-1q8dd3e, .css-1d391kg *, .stTextInput > div > input, .stTextInput > label, .stSelectbox > div, .stSelectbox > label, .stRadio > div, .stRadio > label, .stButton > button, .stFileUploader > div, .stFileUploader > label, .stAlert, .stAlert > div, .stAlert > span, .stSidebar, .stSidebar * {
            color: #000 !important;
        }
        .main-title, .login-title {
            color: #000 !important;
        }
        .stButton > button {
            background-color: #fff !important;
            color: #000 !important;
            font-weight: bold;
            border: 2px solid #000 !important;
            border-radius: 8px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.logo_image:
        st.markdown(f"""
            <div class="header-container">
                <div class="logo-container">
                    <img src="data:image/png;base64,{st.session_state.logo_image}" width="100">
                </div>
                <h1 class="main-title">Plataforma de Análisis y Predicción</h1>
            </div>
        """, unsafe_allow_html=True)

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
