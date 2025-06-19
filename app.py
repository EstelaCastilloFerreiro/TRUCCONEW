import streamlit as st
st.set_page_config(page_title="TRUCCO", page_icon="🡕", layout="wide")

from dashboard import mostrar_dashboard
import pandas as pd
import base64

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
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
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
    set_background("assets/bg_trucco.png")  # Aplica fondo solo antes del login
    st.image("assets/trucco.png", width=180)
    st.markdown("<div class='login-title'>Acceso a TRUCCO Analytics</div>", unsafe_allow_html=True)
    usuario = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Entrar"):
        if usuario and password:
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
    st.markdown("""
        <div class="header-container">
            <div class="logo-container">
                <img src="data:image/png;base64,{}" width="100">
            </div>
            <h1 class="main-title">Plataforma de Análisis y Predicción</h1>
        </div>
    """.format(
        base64.b64encode(open("assets/trucco.png", "rb").read()).decode()
    ), unsafe_allow_html=True)

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
