import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import re  # Add re import for regex
import os
import joblib
import json
from datetime import datetime, timedelta
import numpy as np
from catboost import Pool
import io

# Import model functions
from modelo import prepare_final_dataset_improved

# Configuración estilo gráfico general (sin líneas de fondo)
sns.set_style("white")
sns.set_context("talk", font_scale=0.9)
plt.rcParams.update({
    "axes.edgecolor": "#E0E0E0",
    "axes.linewidth": 0.8,
    "axes.titlesize": 14,
    "axes.titleweight": 'bold',
    "axes.labelcolor": "#333333",
    "axes.labelsize": 12,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "sans-serif"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "figure.autolayout": True,
    "figure.constrained_layout.use": True
})

# Paletas de colores personalizadas
COLOR_GRADIENT = ["#e6f3ff", "#cce7ff", "#99cfff", "#66b8ff", "#33a0ff", "#0088ff", "#006acc", "#004d99", "#003366"]
COLOR_GRADIENT_WARM = ["#fff5e6", "#ffebcc", "#ffd699", "#ffc266", "#ffad33", "#ff9900", "#cc7a00", "#995c00", "#663d00"]
COLOR_GRADIENT_GREEN = ["#e6ffe6", "#ccffcc", "#99ff99", "#66ff66", "#33ff33", "#00ff00", "#00cc00", "#009900", "#006600"]

TIENDAS_EXTRANJERAS = [
    "I301COINBERGAMO(TRUCCO)", "I302COINVARESE(TRUCCO)", "I303COINBARICASAMASSIMA(TRUCCO)",
    "I304COINMILANO5GIORNATE(TRUCCO)", "I305COINROMACINECITTA(TRUCCO)", "I306COINGENOVA(TRUCCO)",
    "I309COINSASSARI(TRUCCO)", "I314COINCATANIA(TRUCCO)", "I315COINCAGLIARI(TRUCCO)",
    "I316COINLECCE(TRUCCO)", "I317COINMILANOCANTORE(TRUCCO)", "I318COINMESTRE(TRUCCO)",
    "I319COINPADOVA(TRUCCO)", "I320COINFIRENZE(TRUCCO)", "I321COINROMASANGIOVANNI(TRUCCO)",
    "TRUCCOONLINEB2C"
]

COL_ONLINE = '#2ca02c'   # verde fuerte
COL_OTRAS = '#ff7f0e'    # naranja

def custom_sort_key(talla):
    """
    Clave de ordenación personalizada para tallas.
    Prioriza: 1. Tallas numéricas, 2. Tallas de letra estándar, 3. Tallas únicas, 4. Resto.
    """
    talla_str = str(talla).upper()
    
    # Prioridad 1: Tallas numéricas (e.g., '36', '38')
    if talla_str.isdigit():
        return (0, int(talla_str))
    
    # Prioridad 2: Tallas de letra estándar
    size_order = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    if talla_str in size_order:
        return (1, size_order.index(talla_str))
        
    # Prioridad 3: Tallas únicas
    if talla_str in ['U', 'ÚNICA', 'UNICA', 'TU']:
        return (2, talla_str)
        
    # Prioridad 4: Resto, ordenado alfabéticamente
    return (3, talla_str)

def setup_streamlit_styles():
    """Configurar estilos de Streamlit"""
    st.markdown("""
    <style>
    .dashboard-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .kpi-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
        width: 100%;
        margin-top: 0;
        padding-top: 0;
    }
    .kpi-row {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        flex-wrap: nowrap;
        width: 100%;
    }
    .kpi-group {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        margin-top: 0;
        background-color: white;
        width: 100%;
    }
    .kpi-group-title {
        color: #666666;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
        margin-top: 0;
        padding-bottom: 5px;
        border-bottom: 1px solid #e5e7eb;
    }
    .kpi-item {
        flex: 1;
        text-align: center;
        padding: 15px;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background-color: white;
        min-width: 150px;
    }
    .small-font {
        color: #666666;
        font-size: 14px;
        margin-bottom: 5px;
        margin-top: 0;
    }
    .metric-value {
        color: #111827;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
    }
    .section-title {
        color: #111827;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 20px;
        margin-top: 0;
        line-height: 1.2;
    }
    .viz-title {
        color: #111827;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
        margin-top: 0;
        line-height: 1.2;
    }
    .viz-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
        margin-top: 20px;
    }
    .viz-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 20px;
        background-color: white;
        height: 100%;
    }
    div.block-container {
        padding-top: 0;
        margin-top: 0;
    }
    div.stMarkdown {
        margin-top: 0;
        padding-top: 0;
    }
    </style>
    """, unsafe_allow_html=True)

def viz_title(text):
    """Función unificada para títulos de visualizaciones"""
    st.markdown(f'<h3 class="viz-title">{text}</h3>', unsafe_allow_html=True)

def titulo(text):
    st.markdown(f"<h4 style='text-align:left;color:#666666;margin:0;padding:0;font-size:20px;font-weight:bold;'>{text}</h4>", unsafe_allow_html=True)

def subtitulo(text):
    st.markdown(f"<h5 style='text-align:left;color:#666666;margin:0;padding:0;font-size:22px;font-weight:bold;'>{text}</h5>", unsafe_allow_html=True)

def aplicar_filtros(df):
    if not pd.api.types.is_datetime64_any_dtype(df['Fecha Documento']):
        df['Fecha Documento'] = pd.to_datetime(df['Fecha Documento'], format='%d/%m/%Y', errors='coerce')
    fecha_min, fecha_max = df['Fecha Documento'].min(), df['Fecha Documento'].max()

    st.sidebar.header("Filtros")
    fecha_inicio, fecha_fin = st.sidebar.date_input(
        "Rango de fechas",
        [fecha_min, fecha_max],
        min_value=fecha_min,
        max_value=fecha_max
    )

    if fecha_inicio > fecha_fin:
        st.sidebar.error("La fecha de inicio debe ser anterior a la fecha de fin.")
        return df.iloc[0:0]

    df_filtrado = df[(df['Fecha Documento'] >= pd.to_datetime(fecha_inicio)) &
                     (df['Fecha Documento'] <= pd.to_datetime(fecha_fin))]
    tiendas = sorted(df_filtrado['NombreTPV'].dropna().unique())
    modo_tienda = st.sidebar.selectbox(
        "Modo selección tiendas",
        ["Todas las tiendas", "Seleccionar tiendas específicas"]
    )
    if modo_tienda == "Todas las tiendas":
        tienda_seleccionada = tiendas
    else:
        tienda_seleccionada = st.sidebar.multiselect(
            "Selecciona tienda(s)",
            options=tiendas
        )
        if not tienda_seleccionada:
            st.sidebar.warning("Selecciona al menos una tienda para mostrar datos.")
            return df.iloc[0:0]
    
    df_filtrado = df_filtrado[df_filtrado['NombreTPV'].isin(tienda_seleccionada)]
    
    return df_filtrado

def create_resizable_chart(chart_key, chart_function):
    """
    Crea un contenedor para el gráfico con funcionalidad de redimensionamiento
    """
    col1, col2 = st.columns([4, 1])
    with col1:
        size = st.select_slider(
            f'Ajustar tamaño del gráfico {chart_key}',
            options=['Pequeño', 'Mediano', 'Grande', 'Extra Grande'],
            value='Mediano',
            key=f'size_{chart_key}'
        )
    
    sizes = {
        'Pequeño': 300,
        'Mediano': 500,
        'Grande': 700,
        'Extra Grande': 900
    }
    
    height = sizes[size]
    
    st.markdown(f'<div class="chart-container" style="height: {height}px;">', unsafe_allow_html=True)
    chart_function(height)
    st.markdown('</div>', unsafe_allow_html=True)

def plot_bar(df, x, y, title, palette='Greens', rotate_x=30, color=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if color:
        sns.barplot(x=x, y=y, data=df, color=color, ax=ax)
    else:
        # Normalizar los valores para el degradado
        norm_values = (df[y] - df[y].min()) / (df[y].max() - df[y].min())
        colors = [COLOR_GRADIENT[int(v * (len(COLOR_GRADIENT)-1))] if not pd.isna(v) else COLOR_GRADIENT[0] for v in norm_values]
        
        sns.barplot(x=x, y=y, data=df, palette=colors, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', color="#111827", loc="left", pad=0)
    ax.set_xlabel(x, fontsize=13)
    ax.set_ylabel(y, fontsize=13)
    plt.xticks(rotation=rotate_x, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    ax.grid(False)
    ax.set_axisbelow(True)
    sns.despine()
    
    # Ajustar valores sobre las barras
    for bar in ax.patches:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.01 * value),
            f'{int(value)}',
            ha='center',
            va='bottom',
            fontsize=10,
            color='#333'
        )
    
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)

def viz_container(title, render_function):
    """Contenedor para visualizaciones"""
    st.markdown('<div class="viz-container">', unsafe_allow_html=True)
    viz_title(title)
    render_function()
    st.markdown('</div>', unsafe_allow_html=True)

def mostrar_dashboard(df_productos, df_traspasos, df_ventas, seccion):
    setup_streamlit_styles()
    
    # Preparar df_ventas con todas las columnas necesarias
    df_ventas = df_ventas.copy()
    
    # Asegurar columna 'Tienda' en ventas
    if "Tienda" not in df_ventas.columns and "NombreTPV" in df_ventas.columns:
        df_ventas["Tienda"] = df_ventas["NombreTPV"]
    
    # Selección de columna de unidades en ventas
    if "Cantidad" not in df_ventas.columns:
        for col in ["Unidades", "Uds", "Qty"]:
            if col in df_ventas.columns:
                df_ventas["Cantidad"] = df_ventas[col]
                break
    
    # Asegurar columna 'Talla' en ventas
    if "Talla" not in df_ventas.columns:
        for col in ["Talla", "Size"]:
            if col in df_ventas.columns:
                df_ventas["Talla"] = df_ventas[col]
                break
    
    # Crear columna 'Ventas Dinero' basada en Subtotal o P.V.P. * Cantidad
    if 'Ventas Dinero' not in df_ventas.columns:
        if 'Subtotal' in df_ventas.columns:
            df_ventas['Ventas Dinero'] = pd.to_numeric(df_ventas['Subtotal'], errors='coerce').fillna(0)
        elif 'P.V.P.' in df_ventas.columns and 'Cantidad' in df_ventas.columns:
            pvp = pd.to_numeric(df_ventas['P.V.P.'], errors='coerce').fillna(0)
            cantidad = pd.to_numeric(df_ventas['Cantidad'], errors='coerce').fillna(0)
            df_ventas['Ventas Dinero'] = pvp * cantidad
        else:
            # Fallback: usar Cantidad como Ventas Dinero si no hay otras opciones
            df_ventas['Ventas Dinero'] = pd.to_numeric(df_ventas['Cantidad'], errors='coerce').fillna(0)
    
    # Crear columna 'Familia' si no existe
    if 'Familia' not in df_ventas.columns:
        if 'Descripción Familia' in df_ventas.columns:
            df_ventas['Familia'] = df_ventas['Descripción Familia'].fillna("Sin Familia")
        else:
            df_ventas['Familia'] = "Sin Familia"
    
    # Crear columna 'Es_Online' para identificar tiendas online
    if 'Es_Online' not in df_ventas.columns:
        df_ventas['Es_Online'] = df_ventas['NombreTPV'].str.contains('ONLINE', case=False, na=False)
    
    # Crear columna 'Mes' si no existe
    if 'Mes' not in df_ventas.columns and 'Fecha Documento' in df_ventas.columns:
        df_ventas['Fecha Documento'] = pd.to_datetime(df_ventas['Fecha Documento'], errors='coerce')
        df_ventas['Mes'] = df_ventas['Fecha Documento'].dt.to_period('M').astype(str)
    
    # Asegurar que las columnas numéricas sean del tipo correcto
    df_ventas['Cantidad'] = pd.to_numeric(df_ventas['Cantidad'], errors='coerce').fillna(0)
    df_ventas['Ventas Dinero'] = pd.to_numeric(df_ventas['Ventas Dinero'], errors='coerce').fillna(0)

    if seccion == "Resumen General":
        try:
            # Calcular KPIs
            total_ventas_dinero = df_ventas['Ventas Dinero'].sum()
            total_familias = df_ventas['Familia'].nunique()
            
            # Separar tiendas físicas y online
            ventas_fisicas = df_ventas[~df_ventas['Es_Online']]
            ventas_online = df_ventas[df_ventas['Es_Online']]
            
            # Calcular KPIs por tipo de tienda
            ventas_fisicas_dinero = ventas_fisicas['Ventas Dinero'].sum()
            ventas_online_dinero = ventas_online['Ventas Dinero'].sum()
            tiendas_fisicas = ventas_fisicas['NombreTPV'].nunique()
            tiendas_online = ventas_online['NombreTPV'].nunique()

            # KPIs Generales en una sola fila
            st.markdown("""
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                    <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                        KPIs Generales
                    </div>
                    <div style="display: flex; justify-content: space-between; gap: 15px;">
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Total Ventas Netas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Número de Familias</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Total Tiendas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                    </div>
                </div>
            """.format(total_ventas_dinero, total_familias, tiendas_fisicas + tiendas_online), unsafe_allow_html=True)
            
            # KPIs por Tipo de Tienda en una sola fila
            st.markdown("""
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                    <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                        KPIs por Tipo de Tienda
                    </div>
                    <div style="display: flex; justify-content: space-between; gap: 15px;">
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tiendas Físicas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Ventas Físicas</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tiendas Online</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{}</p>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Ventas Online</p>
                            <p style="color: #111827; font-size: 24px; font-weight: bold; margin: 0;">{:,.2f}€</p>
                        </div>
                    </div>
                </div>
            """.format(tiendas_fisicas, ventas_fisicas_dinero, tiendas_online, ventas_online_dinero), unsafe_allow_html=True)
            
            # ===== KPIs de Rotación de Stock =====
            st.markdown("### 📊 **KPIs de Rotación de Stock (V2025 e I2025)**")
            
            # Filtrar ventas solo para V2025 e I2025
            ventas_rotacion = df_ventas[df_ventas['Temporada'].isin(['V2025', 'I2025'])].copy()
            
            # Preparar datos de entrada en almacén y traspasos para el cálculo de rotación
            if not df_productos.empty and 'Fecha REAL entrada en almacén' in df_productos.columns:
                # Crear ACT_14 en df_productos para matching
                df_productos_rotacion = df_productos.copy()
                # Clean ACT column by stripping whitespace
                df_productos_rotacion['ACT'] = df_productos_rotacion['ACT'].astype(str).str.strip()
                df_productos_rotacion['ACT_14'] = df_productos_rotacion['ACT'].astype(str).str[:14]
                df_productos_rotacion['Fecha REAL entrada en almacén'] = pd.to_datetime(
                    df_productos_rotacion['Fecha REAL entrada en almacén'], errors='coerce')
                # Clean Talla column by stripping whitespace
                df_productos_rotacion['Talla'] = df_productos_rotacion['Talla'].astype(str).str.strip()
                # Drop rows with missing merge keys
                df_productos_rotacion = df_productos_rotacion.dropna(subset=['ACT_14', 'Talla'])
                print('Productos:', df_productos_rotacion.shape)
                
                # Preparar traspasos
                df_traspasos_rotacion = df_traspasos.copy()
                df_traspasos_rotacion['ACT'] = df_traspasos_rotacion['ACT'].astype(str).str.strip()
                df_traspasos_rotacion['ACT_14'] = df_traspasos_rotacion['ACT'].astype(str).str[:14]
                # Clean Talla column by stripping whitespace
                df_traspasos_rotacion['Talla'] = df_traspasos_rotacion['Talla'].astype(str).str.strip()
                # Drop rows with missing merge keys
                df_traspasos_rotacion = df_traspasos_rotacion.dropna(subset=['ACT_14', 'Talla', 'Tienda'])
                print('Traspasos:', df_traspasos_rotacion.shape)
                
                # Use robust column finding function
                fecha_enviado_col = get_fecha_enviado_column(df_traspasos_rotacion)
                
                if fecha_enviado_col is not None:
                    df_traspasos_rotacion['Fecha Enviado'] = pd.to_datetime(
                        df_traspasos_rotacion[fecha_enviado_col], errors='coerce')
                    
                    # Preparar ventas
                    ventas_rotacion['ACT'] = ventas_rotacion['ACT'].astype(str).str.strip()
                    ventas_rotacion['ACT_14'] = ventas_rotacion['ACT'].astype(str).str[:14]
                    ventas_rotacion['Fecha Documento'] = pd.to_datetime(
                        ventas_rotacion['Fecha Documento'], errors='coerce')
                    # Clean Talla column by stripping whitespace
                    ventas_rotacion['Talla'] = ventas_rotacion['Talla'].astype(str).str.strip()
                    # Drop rows with missing merge keys
                    ventas_rotacion = ventas_rotacion.dropna(subset=['ACT_14', 'Talla', 'NombreTPV'])
                    print('Ventas:', ventas_rotacion.shape)
                    
                    # Debug: Show sample values to understand merge failure
                    print('Sample ACT_14 ventas:', ventas_rotacion['ACT_14'].head().tolist())
                    print('Sample ACT_14 productos:', df_productos_rotacion['ACT_14'].head().tolist())
                    print('Sample Talla ventas:', ventas_rotacion['Talla'].head().tolist())
                    print('Sample Talla productos:', df_productos_rotacion['Talla'].head().tolist())
                    
                    # OPTIMIZACIÓN: Usar merge en lugar de loops anidados
                    # 1. Merge ventas con entrada en almacén
                    ventas_con_entrada = ventas_rotacion.merge(
                        df_productos_rotacion[['ACT_14', 'Talla', 'Fecha REAL entrada en almacén']],
                        on=['ACT_14', 'Talla'],
                        how='inner'
                    )
                    print('Merge ventas-productos:', ventas_con_entrada.shape)
                    
                    # 2. Merge con traspasos
                    rotacion_completa = ventas_con_entrada.merge(
                        df_traspasos_rotacion[['ACT_14', 'Talla', 'Tienda', 'Fecha Enviado']],
                        left_on=['ACT_14', 'Talla', 'NombreTPV'],
                        right_on=['ACT_14', 'Talla', 'Tienda'],
                        how='inner'
                    )
                    print('Merge completo (rotación):', rotacion_completa.shape)
                    
                    # 3. Calcular días de rotación
                    rotacion_completa['Dias_Rotacion'] = (
                        rotacion_completa['Fecha Documento'] - rotacion_completa['Fecha REAL entrada en almacén']
                    ).dt.days
                    
                    # Filtrar solo días positivos
                    rotacion_completa = rotacion_completa[rotacion_completa['Dias_Rotacion'] >= 0]
                    
                    if not rotacion_completa.empty:
                        # Calcular rotación por tienda
                        rotacion_por_tienda = rotacion_completa.groupby('NombreTPV').agg({
                            'Dias_Rotacion': ['mean', 'count']
                        }).reset_index()
                        rotacion_por_tienda.columns = ['Tienda', 'Dias_Promedio', 'Productos_Con_Rotacion']
                        
                        # Calcular rotación por producto
                        rotacion_por_producto = rotacion_completa.groupby(['ACT_14', 'Descripción Familia']).agg({
                            'Dias_Rotacion': ['mean', 'count']
                        }).reset_index()
                        rotacion_por_producto.columns = ['ACT', 'Producto', 'Dias_Promedio', 'Ventas_Con_Rotacion']
                        
                        # Calcular KPIs
                        tienda_mayor_rotacion = "Sin datos"
                        tienda_mayor_rotacion_dias = 0
                        tienda_menor_rotacion = "Sin datos"
                        tienda_menor_rotacion_dias = 0
                        producto_mayor_rotacion = "Sin datos"
                        producto_mayor_rotacion_dias = 0
                        producto_menor_rotacion = "Sin datos"
                        producto_menor_rotacion_dias = 0
                        
                        if not rotacion_por_tienda.empty:
                            # Tienda con mayor rotación (menos días)
                            idx_mayor = rotacion_por_tienda['Dias_Promedio'].idxmin()
                            tienda_mayor_rotacion = rotacion_por_tienda.loc[idx_mayor, 'Tienda']
                            tienda_mayor_rotacion_dias = rotacion_por_tienda.loc[idx_mayor, 'Dias_Promedio']
                            
                            # Tienda con menor rotación (más días)
                            idx_menor = rotacion_por_tienda['Dias_Promedio'].idxmax()
                            tienda_menor_rotacion = rotacion_por_tienda.loc[idx_menor, 'Tienda']
                            tienda_menor_rotacion_dias = rotacion_por_tienda.loc[idx_menor, 'Dias_Promedio']
                        
                        if not rotacion_por_producto.empty:
                            # Producto con mayor rotación (menos días)
                            idx_mayor = rotacion_por_producto['Dias_Promedio'].idxmin()
                            producto_mayor_rotacion = rotacion_por_producto.loc[idx_mayor, 'Producto']
                            producto_mayor_rotacion_dias = rotacion_por_producto.loc[idx_mayor, 'Dias_Promedio']
                            
                            # Producto con menor rotación (más días)
                            idx_menor = rotacion_por_producto['Dias_Promedio'].idxmax()
                            producto_menor_rotacion = rotacion_por_producto.loc[idx_menor, 'Producto']
                            producto_menor_rotacion_dias = rotacion_por_producto.loc[idx_menor, 'Dias_Promedio']
                        
                        # Mostrar KPIs de rotación
                        st.markdown("""
                            <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: white;">
                                <div style="color: #666666; font-size: 16px; font-weight: 600; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid #e5e7eb;">
                                    KPIs de Rotación de Stock
                                </div>
                                <div style="display: flex; justify-content: space-between; gap: 15px;">
                                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tienda Mayor Rotación</p>
                                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                        <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f} días promedio</p>
                                    </div>
                                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Tienda Menor Rotación</p>
                                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                        <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.1f} días promedio</p>
                                    </div>
                                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Producto Mayor Rotación</p>
                                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                        <p style="color: #059669; font-size: 12px; margin: 0;">{:.1f} días promedio</p>
                                    </div>
                                    <div style="flex: 1; text-align: center; padding: 15px; border: 1px solid #e5e7eb; border-radius: 8px; background-color: white;">
                                        <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Producto Menor Rotación</p>
                                        <p style="color: #111827; font-size: 18px; font-weight: bold; margin: 0;">{}</p>
                                        <p style="color: #dc2626; font-size: 12px; margin: 0;">{:.1f} días promedio</p>
                                    </div>
                                </div>
                            </div>
                        """.format(
                            tienda_mayor_rotacion, tienda_mayor_rotacion_dias,
                            tienda_menor_rotacion, tienda_menor_rotacion_dias,
                            producto_mayor_rotacion, producto_mayor_rotacion_dias,
                            producto_menor_rotacion, producto_menor_rotacion_dias
                        ), unsafe_allow_html=True)
                        
                        # Mostrar estadísticas adicionales
                        st.info(f"📊 Análisis basado en {len(rotacion_completa)} productos con rotación calculada")
                    else:
                        st.info("No se encontraron productos con datos completos de entrada, traspaso y venta para calcular rotación.")
                else:
                    st.warning("⚠️ No se encontró la columna 'Fecha Enviado' en los datos de traspasos. Saltando cálculo de rotación.")
            else:
                st.info("No hay datos de entrada en almacén disponibles para calcular rotación de stock.")

            # Col 1: Ventas por mes (centered)
            col1a, col1b, col1c = st.columns([1, 2, 1])
            
            with col1b:
                viz_title("Ventas Mensuales por Tipo de Tienda")
                ventas_mes_tipo = df_ventas.groupby(['Mes', 'Es_Online']).agg({
                    'Cantidad': 'sum',
                    'Ventas Dinero': 'sum'
                }).reset_index()
                
                ventas_mes_tipo['Tipo'] = ventas_mes_tipo['Es_Online'].map({True: 'Online', False: 'Física'})
                
                fig = px.bar(ventas_mes_tipo, 
                            x='Mes', 
                            y='Cantidad', 
                            color='Tipo',
                            color_discrete_map={'Física': '#1e3a8a', 'Online': '#60a5fa'},
                            barmode='stack',
                            text='Cantidad',
                            height=400)
                
                fig.update_layout(
                    xaxis_title="Mes",
                    yaxis_title="Cantidad",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                fig.update_traces(
                    texttemplate='%{text:,.0f}', 
                    textposition='outside',
                    hovertemplate="Mes: %{x}<br>Cantidad: %{text:,.0f}<br>Ventas: %{customdata:,.2f}€<extra></extra>",
                    customdata=ventas_mes_tipo['Ventas Dinero'],
                    opacity=0.8
                )
                
                st.markdown(f"""
                    <div style="width: 100%;">
                """, unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=False)
                st.markdown("</div>", unsafe_allow_html=True)

            # Col 2,3: Ranking de tiendas
            col2, col3 = st.columns(2)
            
            if tiendas_especificas:
                # Mostrar tabla de ranking para las tiendas seleccionadas (full width)
                viz_title("Ranking de Tiendas Seleccionadas")
                
                # Filtrar solo las tiendas seleccionadas del ranking completo
                tiendas_ranking = ventas_por_tienda_completo[ventas_por_tienda_completo['Tienda'].isin(tienda_seleccionada)].copy()
                
                # Calcular la familia más vendida para cada tienda (cached)
                familias_por_tienda = calculate_family_rankings(df_ventas)
                
                # Obtener la familia top para cada tienda seleccionada
                familias_top = []
                for tienda in tiendas_ranking['Tienda']:
                    familias_tienda = familias_por_tienda[familias_por_tienda['NombreTPV'] == tienda]
                    if not familias_tienda.empty:
                        familia_top = familias_tienda.iloc[0]['Familia']
                        familias_top.append(familia_top)
                    else:
                        familias_top.append('Sin datos')
                
                tiendas_ranking['Familia Top'] = familias_top
                
                # Reordenar columnas
                tiendas_ranking = tiendas_ranking[['Tienda', 'Ranking', 'Unidades Vendidas', 'Ventas (€)', 'Familia Top']]
                
                # Mostrar tabla
                st.dataframe(
                    tiendas_ranking.style.format({
                        'Unidades Vendidas': '{:,.0f}',
                        'Ventas (€)': '{:,.2f}€'
                    }),
                    use_container_width=True
                )
            else:
                # Mostrar top 20 y bottom 20 como antes
                with col2:
                    # Top 20 tiendas con más ventas por ventas (€)
                    viz_title("Top 20 tiendas con más ventas")
                    top_20_tiendas = ventas_por_tienda_completo.head(20)
                    
                    fig = px.bar(
                        top_20_tiendas,
                        x='Tienda',
                        y='Ventas (€)',
                        color='Ventas (€)',
                        color_continuous_scale=COLOR_GRADIENT,
                        height=400,
                        labels={'Tienda': 'Tienda', 'Ventas (€)': 'Ventas (€)', 'Unidades Vendidas': 'Unidades'}
                    )
                    fig.update_layout(
                        xaxis_tickangle=45,
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    fig.update_traces(
                        texttemplate='%{y:,.2f}€',
                        textposition='outside',
                        hovertemplate="Tienda: %{x}<br>Ventas: %{y:,.2f}€<br>Unidades: %{customdata:,}<extra></extra>",
                        customdata=top_20_tiendas['Unidades Vendidas'],
                        opacity=0.8
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col3:
                    # Top 20 tiendas con menos ventas por ventas (€)
                    viz_title("Top 20 tiendas con menos ventas")
                    bottom_20_tiendas = ventas_por_tienda_completo.tail(20)
                    
                    fig = px.bar(
                        bottom_20_tiendas,
                        x='Tienda',
                        y='Ventas (€)',
                        color='Ventas (€)',
                        color_continuous_scale=COLOR_GRADIENT,
                        height=400,
                        labels={'Tienda': 'Tienda', 'Ventas (€)': 'Ventas (€)', 'Unidades Vendidas': 'Unidades'}
                    )
                    fig.update_layout(
                        xaxis_tickangle=45,
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    fig.update_traces(
                        texttemplate='%{y:,.2f}€',
                        textposition='outside',
                        hovertemplate="Tienda: %{x}<br>Ventas: %{y:,.2f}€<br>Unidades: %{customdata:,}<extra></extra>",
                        customdata=bottom_20_tiendas['Unidades Vendidas'],
                        opacity=0.8
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Col 4: Unidades Vendidas por Talla (centered)
            col4a, col4b, col4c = st.columns([1, 2, 1])
            
            with col4b:
                viz_title("Unidades Vendidas por Talla")
                
                familias = sorted(df_ventas['Familia'].unique())
                familia_seleccionada = st.selectbox("Selecciona una familia:", familias)
                
                # Filtrar df_ventas por familia seleccionada
                df_familia = df_ventas[df_ventas['Familia'] == familia_seleccionada].copy()

                if df_familia.empty:
                    st.warning("No hay datos de ventas para la familia seleccionada.")
                else:
                    # Agrupamos por Talla y Temporada
                    tallas_sumadas = (
                        df_familia.groupby(['Talla', 'Temporada'])['Cantidad']
                        .sum()
                        .reset_index()
                    )
                    
                    # Orden personalizado de tallas
                    tallas_presentes = df_familia['Talla'].dropna().unique()
                    tallas_orden = sorted(tallas_presentes, key=custom_sort_key)

                    # Gráfico de barras apiladas por Temporada
                    temporada_colors = get_temporada_colors(df_ventas)
                    fig = px.bar(
                        tallas_sumadas,
                        x='Talla',
                        y='Cantidad',
                        color='Temporada',
                        text='Cantidad',
                        category_orders={'Talla': tallas_orden},
                        color_discrete_map=temporada_colors,
                        height=450
                    )
                    
                    fig.update_layout(
                        xaxis_title="Talla",
                        yaxis_title="Unidades Vendidas",
                        barmode="stack",
                        margin=dict(t=30, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    
                    fig.update_traces(texttemplate='%{text:.0f}', textposition='inside', opacity=0.9)
                    st.plotly_chart(fig, use_container_width=True)



            # Col 5,6: Tablas por Temporada con layout dinámico
            # Preparar datos de entrada en almacén para las tablas por temporada
            # Agregar Descripción Familia a df_productos usando ACT codes de df_ventas
            df_productos_temp = df_productos.copy()
            

            
            # Crear ACT_14 (primeros 14 caracteres) en df_productos
            df_productos_temp['ACT_14'] = df_productos_temp['ACT'].astype(str).str[:14]
            
            # Crear mapeo de ACT a Descripción Familia desde df_ventas
            act_to_familia = df_ventas[['ACT', 'Descripción Familia']].drop_duplicates().set_index('ACT')['Descripción Familia'].to_dict()
            
            # Agregar Descripción Familia a df_productos usando ACT_14
            df_productos_temp['Descripción Familia'] = df_productos_temp['ACT_14'].map(act_to_familia)
            
            # Filtrar por la familia seleccionada
            df_almacen_fam = df_productos_temp[
                df_productos_temp['Descripción Familia'] == familia_seleccionada
            ].copy()
            
            # Identificar productos sin familia asignada
            df_sin_familia = df_productos_temp[
                df_productos_temp['Descripción Familia'].isna()
            ].copy()
            
            # Inicializar df_pendientes como DataFrame vacío
            df_pendientes = pd.DataFrame()
            

            
            if not df_almacen_fam.empty and 'Fecha REAL entrada en almacén' in df_almacen_fam.columns:
                # Convertir fecha de entrada en almacén
                df_almacen_fam['Fecha REAL entrada en almacén'] = pd.to_datetime(
                    df_almacen_fam['Fecha REAL entrada en almacén'], errors='coerce')
                
                # Separar filas con fecha válida y sin fecha
                df_almacen_fam_con_fecha = df_almacen_fam.dropna(subset=['Fecha REAL entrada en almacén'])
                df_almacen_fam_sin_fecha = df_almacen_fam[df_almacen_fam['Fecha REAL entrada en almacén'].isna()].copy()
                
                # Agregar mes de entrada para filas con fecha válida
                df_almacen_fam_con_fecha['Mes Entrada'] = df_almacen_fam_con_fecha['Fecha REAL entrada en almacén'].dt.to_period('M').astype(str)
                
                # Separar filas pendientes de entrega (sin fecha válida)
                if not df_almacen_fam_sin_fecha.empty:
                    # Crear DataFrame separado para pendientes de entrega
                    df_pendientes = df_almacen_fam_sin_fecha.copy()
                    df_pendientes['Estado'] = 'Pendiente de entrega'
                else:
                    df_pendientes = pd.DataFrame()
                
                # Usar solo las filas con fecha válida para el análisis de almacén
                df_almacen_fam = df_almacen_fam_con_fecha
                
                # Obtener el último mes de df_ventas para filtrar los datos
                ultimo_mes_ventas = df_ventas['Mes'].max()
                
                # Preparar datos para la tabla por Temporada
                # Buscar la columna correcta para cantidad de entrada en almacén
                cantidad_col = None
                for col in df_almacen_fam.columns:
                    if 'cantidad' in col.lower() and 'pedida' not in col.lower():
                        cantidad_col = col
                        break
                
                if cantidad_col is None:
                    # Si no encontramos una columna específica, usar 'Cantidad' o la primera columna numérica
                    numeric_cols = df_almacen_fam.select_dtypes(include=[np.number]).columns
                    if 'Cantidad' in df_almacen_fam.columns:
                        cantidad_col = 'Cantidad'
                    elif len(numeric_cols) > 0:
                        cantidad_col = numeric_cols[0]
                    else:
                        st.error("No se encontró una columna de cantidad válida")
                        cantidad_col = 'Cantidad'  # Fallback
                

                
                datos_tabla = (
                    df_almacen_fam.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                    .sum()
                    .reset_index()
                    .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                    .sort_values(['Mes Entrada', 'Talla'])
                )
                
                # Filtrar datos hasta el último mes de ventas
                datos_tabla = datos_tabla[datos_tabla['Mes Entrada'] <= ultimo_mes_ventas]
                
                if not datos_tabla.empty:
                    # Crear Tema_6 (primeros 6 caracteres) en df_almacen_fam
                    df_almacen_fam['Tema_6'] = df_almacen_fam['Tema'].astype(str).str[:6]
                    
                    
                    # Obtener todos los temas únicos de df_productos (no solo los vendidos)
                    temas_productos = sorted(df_almacen_fam['Tema_6'].unique())
                    
                    # Calcular temas y num_temas SIEMPRE
                    temas = temas_productos
                    num_temas = len(temas)

                    if num_temas > 0:
                        # --- Sección: Entradas almacén y traspasos ---
                        st.markdown('<hr style="margin: 1em 0; border-top: 2px solid #bbb;">', unsafe_allow_html=True)
                        st.markdown('<h4 style="color:#333;font-weight:bold;">Entradas almacén y traspasos</h4>', unsafe_allow_html=True)
                        
                        # Si se han seleccionado tiendas específicas, mostrar tabla de análisis temporal
                        if tiendas_especificas:
                            st.subheader("Análisis Temporal: Entrada Almacén → Envío → Primera Venta")
                            # Preparar datos para el análisis temporal
                            timeline_data = []
                            df_almacen_fam_timeline = df_almacen_fam.copy()
                            df_almacen_fam_timeline['Fecha REAL entrada en almacén'] = pd.to_datetime(
                                df_almacen_fam_timeline['Fecha REAL entrada en almacén'], errors='coerce')
                            df_traspasos_timeline = df_traspasos_filtrado.copy()
                            
                            # Use robust column finding function
                            fecha_enviado_col_timeline = get_fecha_enviado_column(df_traspasos_timeline)
                            
                            if fecha_enviado_col_timeline is not None:
                                df_traspasos_timeline['Fecha Enviado'] = pd.to_datetime(
                                    df_traspasos_timeline[fecha_enviado_col_timeline], errors='coerce')
                                
                                # Create ACT_14 in traspasos data to match warehouse data
                                df_traspasos_timeline['ACT_14'] = df_traspasos_timeline['ACT'].astype(str).str[:14]
                                df_ventas_timeline = df_ventas.copy()
                                
                                df_ventas_timeline['Fecha Documento'] = pd.to_datetime(
                                    df_ventas_timeline['Fecha Documento'], errors='coerce')
                                
                                merged = pd.merge(
                                    df_almacen_fam_timeline,
                                    df_traspasos_timeline,
                                    left_on=['ACT_14', 'Talla'],
                                    right_on=['ACT_14', 'Talla'],
                                    suffixes=('_almacen', '_traspaso')
                                )
                                merged = merged[merged['Fecha Enviado'] >= merged['Fecha REAL entrada en almacén']]
                                
                                for _, row in merged.iterrows():
                                    fecha_entrada = row['Fecha REAL entrada en almacén']
                                    fecha_envio = row['Fecha Enviado']
                                    act = row['ACT_almacen'].strip()  # Remove trailing spaces
                                    talla = row['Talla'].strip()  # Remove trailing spaces
                                    tienda_envio = row['Tienda']
                                    tema = row['Tema']
                                    
                                    ventas_producto = df_ventas_timeline[
                                        (df_ventas_timeline['ACT'].str.strip() == act) &
                                        (df_ventas_timeline['Talla'].str.strip() == talla) &
                                        (df_ventas_timeline['NombreTPV'].str.strip() == tienda_envio.strip()) &
                                        (df_ventas_timeline['Fecha Documento'] >= fecha_entrada) &
                                        (df_ventas_timeline['Cantidad'] > 0)
                                    ]
                                    
                                    if not ventas_producto.empty:
                                        primera_venta = ventas_producto.loc[ventas_producto['Fecha Documento'].idxmin()]
                                        fecha_primera_venta = primera_venta['Fecha Documento']
                                        dias_entrada_venta = (fecha_primera_venta - fecha_entrada).days
                                    else:
                                        fecha_primera_venta = None
                                        dias_entrada_venta = -1  # Use -1 instead of None for "Sin ventas"
                                    dias_entrada_envio = (fecha_envio - fecha_entrada).days
                                    timeline_data.append({
                                        'ACT': act,
                                        'Tema': tema,
                                        'Talla': talla,
                                        'Tienda Envío': tienda_envio,
                                        'Fecha Entrada Almacén': fecha_entrada.strftime('%d/%m/%Y'),
                                        'Fecha Enviado': fecha_envio.strftime('%d/%m/%Y'),
                                        'Fecha Primera Venta': fecha_primera_venta.strftime('%d/%m/%Y') if fecha_primera_venta else "Sin ventas",
                                        'Días Entrada-Envío': dias_entrada_envio,
                                        'Días Entrada-Primera Venta': dias_entrada_venta if dias_entrada_venta != -1 else -1
                                    })
                                
                                if timeline_data:
                                    df_timeline = pd.DataFrame(timeline_data)
                                    df_timeline['Fecha Entrada Almacén'] = pd.to_datetime(df_timeline['Fecha Entrada Almacén'], format='%d/%m/%Y')
                                    df_timeline = df_timeline.sort_values('Fecha Entrada Almacén', ascending=False)
                                    df_timeline['Fecha Entrada Almacén'] = df_timeline['Fecha Entrada Almacén'].dt.strftime('%d/%m/%Y')
                                    st.dataframe(
                                        df_timeline,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        avg_dias_envio = pd.to_numeric(df_timeline['Días Entrada-Envío'], errors='coerce').mean()
                                        st.metric("Promedio días Entrada→Envío", f"{avg_dias_envio:.1f} días")
                                    with col2:
                                        avg_dias_venta = pd.to_numeric(df_timeline['Días Entrada-Primera Venta'].replace('Sin ventas', pd.NA), errors='coerce').mean()
                                        st.metric("Promedio días Entrada→Primera Venta", f"{avg_dias_venta:.1f} días" if not pd.isna(avg_dias_venta) else "N/A")
                                    with col3:
                                        total_productos = len(df_timeline)
                                        st.metric("Total productos analizados", f"{total_productos}")
                                else:
                                    st.info("No se encontraron datos de envíos para los productos de entrada en almacén de la familia seleccionada.")
                            else:
                                st.warning("⚠️ No se encontró la columna 'Fecha Enviado' en los datos de traspasos para el análisis temporal.")
                        else:
                            if num_temas == 1:
                                # Un tema: centrado
                                col5a, col5b, col5c = st.columns([1, 2, 1])
                                with col5b:
                                    tema = temas[0]
                                    st.subheader(f"Entrada Almacén - {tema}")
                                    
                                    # Crear gráfico de comparación enviado vs ventas
                                    if tema == 'T_OI25':
                                        temporada_comparacion = 'I2025'
                                    elif tema == 'T_PV25':
                                        temporada_comparacion = 'V2025'
                                    else:
                                        temporada_comparacion = None
                                    
                                    if temporada_comparacion:
                                        ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                        if not ventas_temporada.empty:
                                            act_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]['ACT_14'].unique()
                                            ventas_tema = ventas_temporada[ventas_temporada['ACT'].isin(act_tema)]
                                            if not ventas_tema.empty:
                                                ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                enviado_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                                enviado_por_talla = enviado_tema.groupby('Talla')[cantidad_col].sum().reset_index()
                                                datos_comparacion = pd.merge(
                                                    enviado_por_talla, 
                                                    ventas_por_talla, 
                                                    on='Talla', 
                                                    how='outer'
                                                ).fillna(0)
                                                # Ordenar tallas
                                                datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                fig, ax = plt.subplots(figsize=(10, 6))
                                                x = np.arange(len(datos_comparacion))
                                                width = 0.35
                                                ax.bar(x - width/2, datos_comparacion[cantidad_col], width, label='Enviado Almacén', color='purple', alpha=0.8)
                                                ax.bar(x + width/2, datos_comparacion['Cantidad'], width, label='Ventas', color='darkblue', alpha=0.8)
                                                ax.set_xlabel('Talla')
                                                ax.set_ylabel('Cantidad')
                                                ax.set_title(f'Enviado vs Ventas - {tema} ({temporada_comparacion})')
                                                ax.set_xticks(x)
                                                ax.set_xticklabels(datos_comparacion['Talla'])
                                                ax.legend()
                                                ax.grid(True, alpha=0.3)
                                                st.pyplot(fig)
                                                plt.close()
                                    
                                    # Filtrar datos para este tema específico
                                    datos_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                    datos_tabla_tema = (
                                        datos_tema.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                                        .sum()
                                        .reset_index()
                                        .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                                        .sort_values(['Mes Entrada', 'Talla'])
                                    )
                                    
                                    if not datos_tabla_tema.empty:
                                        # Crear tabla pivot para mejor visualización
                                        tabla_pivot = datos_tabla_tema.pivot_table(
                                            index='Mes Entrada',
                                            columns='Talla',
                                            values='Cantidad Entrada Almacén',
                                            fill_value=0
                                        ).round(0)
                                        tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                        tabla_pivot = tabla_pivot[tallas_orden]
                                        st.dataframe(
                                            tabla_pivot.style.format("{:,.0f}"),
                                            use_container_width=True,
                                            hide_index=False
                                        )
                                        total_temp = tabla_pivot.sum().sum()
                                        st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                    else:
                                        st.info(f"No hay datos para el tema {tema}")
                            elif num_temas == 2:
                                col5, col6 = st.columns(2)
                                for i, tema in enumerate(temas):
                                    with locals()[f'col{5+i}']:
                                        st.subheader(f"Entrada Almacén - {tema}")
                                        
                                        # Crear gráfico de comparación enviado vs ventas
                                        if tema == 'T_OI25':
                                            temporada_comparacion = 'I2025'
                                        elif tema == 'T_PV25':
                                            temporada_comparacion = 'V2025'
                                        else:
                                            temporada_comparacion = None
                                        
                                        if temporada_comparacion:
                                            ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                            if not ventas_temporada.empty:
                                                act_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]['ACT_14'].unique()
                                                ventas_tema = ventas_temporada[ventas_temporada['ACT'].isin(act_tema)]
                                                if not ventas_tema.empty:
                                                    ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                    enviado_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                                    enviado_por_talla = enviado_tema.groupby('Talla')[cantidad_col].sum().reset_index()
                                                    datos_comparacion = pd.merge(
                                                        enviado_por_talla, 
                                                        ventas_por_talla, 
                                                        on='Talla', 
                                                        how='outer'
                                                    ).fillna(0)
                                                    # Ordenar tallas
                                                    datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                    fig, ax = plt.subplots(figsize=(8, 5))
                                                    x = np.arange(len(datos_comparacion))
                                                    width = 0.35
                                                    ax.bar(x - width/2, datos_comparacion[cantidad_col], width, label='Enviado Almacén', color='purple', alpha=0.8)
                                                    ax.bar(x + width/2, datos_comparacion['Cantidad'], width, label='Ventas', color='darkblue', alpha=0.8)
                                                    ax.set_xlabel('Talla')
                                                    ax.set_ylabel('Cantidad')
                                                    ax.set_title(f'Enviado vs Ventas - {tema} ({temporada_comparacion})')
                                                    ax.set_xticks(x)
                                                    ax.set_xticklabels(datos_comparacion['Talla'])
                                                    ax.legend()
                                                    ax.grid(True, alpha=0.3)
                                                    st.pyplot(fig)
                                                    plt.close()
                                        
                                        # Filtrar datos para este tema específico
                                        datos_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                        datos_tabla_tema = (
                                            datos_tema.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                                            .sum()
                                            .reset_index()
                                            .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                                            .sort_values(['Mes Entrada', 'Talla'])
                                        )
                                        
                                        if not datos_tabla_tema.empty:
                                            # Crear tabla pivot para mejor visualización
                                            tabla_pivot = datos_tabla_tema.pivot_table(
                                                index='Mes Entrada',
                                                columns='Talla',
                                                values='Cantidad Entrada Almacén',
                                                fill_value=0
                                            ).round(0)
                                            tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                            tabla_pivot = tabla_pivot[tallas_orden]
                                            st.dataframe(
                                                tabla_pivot.style.format("{:,.0f}"),
                                                use_container_width=True,
                                                hide_index=False
                                            )
                                            total_temp = tabla_pivot.sum().sum()
                                            st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                        else:
                                            st.info(f"No hay datos para el tema {tema}")
                            else:
                                col5, col6 = st.columns(2)
                                mitad = (num_temas + 1) // 2
                                temas_col5 = temas[:mitad]
                                temas_col6 = temas[mitad:]
                                with col5:
                                    for tema in temas_col5:
                                        st.subheader(f"Entrada Almacén - {tema}")
                                        
                                        # Crear gráfico de comparación enviado vs ventas
                                        if tema == 'T_OI25':
                                            temporada_comparacion = 'I2025'
                                        elif tema == 'T_PV25':
                                            temporada_comparacion = 'V2025'
                                        else:
                                            temporada_comparacion = None
                                        
                                        if temporada_comparacion:
                                            # Obtener datos de ventas para la temporada
                                            ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                            if not ventas_temporada.empty:
                                                # Obtener ACTs del tema actual
                                                act_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]['ACT_14'].unique()
                                                
                                                # Filtrar ventas por ACTs del tema
                                                ventas_tema = ventas_temporada[ventas_temporada['ACT'].isin(act_tema)]
                                                
                                                if not ventas_tema.empty:
                                                    # Agrupar ventas por talla
                                                    ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                    
                                                    # Obtener datos de enviado del tema
                                                    enviado_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                                    enviado_por_talla = enviado_tema.groupby('Talla')[cantidad_col].sum().reset_index()
                                                    
                                                    # Combinar datos
                                                    datos_comparacion = pd.merge(
                                                        enviado_por_talla, 
                                                        ventas_por_talla, 
                                                        on='Talla', 
                                                        how='outer'
                                                    ).fillna(0)
                                                    
                                                    # Ordenar tallas
                                                    datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                    
                                                    # Crear gráfico
                                                    fig, ax = plt.subplots(figsize=(8, 5))
                                                    
                                                    x = np.arange(len(datos_comparacion))
                                                    width = 0.35
                                                    
                                                    ax.bar(x - width/2, datos_comparacion[cantidad_col], width, label='Enviado Almacén', color='purple', alpha=0.8)
                                                    ax.bar(x + width/2, datos_comparacion['Cantidad'], width, label='Ventas', color='darkblue', alpha=0.8)
                                                    
                                                    ax.set_xlabel('Talla')
                                                    ax.set_ylabel('Cantidad')
                                                    ax.set_title(f'Enviado vs Ventas - {tema} ({temporada_comparacion})')
                                                    ax.set_xticks(x)
                                                    ax.set_xticklabels(datos_comparacion['Talla'])
                                                    ax.legend()
                                                    ax.grid(True, alpha=0.3)
                                                    st.pyplot(fig)
                                                    plt.close()
                                        
                                        # Filtrar datos para este tema específico
                                        datos_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                        datos_tabla_tema = (
                                            datos_tema.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                                            .sum()
                                            .reset_index()
                                            .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                                            .sort_values(['Mes Entrada', 'Talla'])
                                        )
                                        
                                        if not datos_tabla_tema.empty:
                                            # Crear tabla pivot para mejor visualización
                                            tabla_pivot = datos_tabla_tema.pivot_table(
                                                index='Mes Entrada',
                                                columns='Talla',
                                                values='Cantidad Entrada Almacén',
                                                fill_value=0
                                            ).round(0)
                                            tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                            tabla_pivot = tabla_pivot[tallas_orden]
                                            st.dataframe(
                                                tabla_pivot.style.format("{:,.0f}"),
                                                use_container_width=True,
                                                hide_index=False
                                            )
                                            total_temp = tabla_pivot.sum().sum()
                                            st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                        else:
                                            st.info(f"No hay datos para el tema {tema}")
                                with col6:
                                    for tema in temas_col6:
                                        st.subheader(f"Entrada Almacén - {tema}")
                                        
                                        # Crear gráfico de comparación enviado vs ventas
                                        if tema == 'T_OI25':
                                            temporada_comparacion = 'I2025'
                                        elif tema == 'T_PV25':
                                            temporada_comparacion = 'V2025'
                                        else:
                                            temporada_comparacion = None
                                        
                                        if temporada_comparacion:
                                            # Obtener datos de ventas para la temporada
                                            ventas_temporada = df_ventas[df_ventas['Temporada'] == temporada_comparacion]
                                            if not ventas_temporada.empty:
                                                # Obtener ACTs del tema actual
                                                act_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]['ACT_14'].unique()
                                                
                                                # Filtrar ventas por ACTs del tema
                                                ventas_tema = ventas_temporada[ventas_temporada['ACT'].isin(act_tema)]
                                                
                                                if not ventas_tema.empty:
                                                    # Agrupar ventas por talla
                                                    ventas_por_talla = ventas_tema.groupby('Talla')['Cantidad'].sum().reset_index()
                                                    
                                                    # Obtener datos de enviado del tema
                                                    enviado_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                                    enviado_por_talla = enviado_tema.groupby('Talla')[cantidad_col].sum().reset_index()
                                                    
                                                    # Combinar datos
                                                    datos_comparacion = pd.merge(
                                                        enviado_por_talla, 
                                                        ventas_por_talla, 
                                                        on='Talla', 
                                                        how='outer'
                                                    ).fillna(0)
                                                    
                                                    # Ordenar tallas
                                                    datos_comparacion = datos_comparacion.sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                                                    
                                                    # Crear gráfico
                                                    fig, ax = plt.subplots(figsize=(8, 5))
                                                    
                                                    x = np.arange(len(datos_comparacion))
                                                    width = 0.35
                                                    
                                                    ax.bar(x - width/2, datos_comparacion[cantidad_col], width, label='Enviado Almacén', color='purple', alpha=0.8)
                                                    ax.bar(x + width/2, datos_comparacion['Cantidad'], width, label='Ventas', color='darkblue', alpha=0.8)
                                                    
                                                    ax.set_xlabel('Talla')
                                                    ax.set_ylabel('Cantidad')
                                                    ax.set_title(f'Enviado vs Ventas - {tema} ({temporada_comparacion})')
                                                    ax.set_xticks(x)
                                                    ax.set_xticklabels(datos_comparacion['Talla'])
                                                    ax.legend()
                                                    ax.grid(True, alpha=0.3)
                                                    st.pyplot(fig)
                                                    plt.close()
                                        
                                        # Filtrar datos para este tema específico
                                        datos_tema = df_almacen_fam[df_almacen_fam['Tema_6'] == tema]
                                        datos_tabla_tema = (
                                            datos_tema.groupby(['Mes Entrada', 'Talla'])[cantidad_col]
                                            .sum()
                                            .reset_index()
                                            .rename(columns={cantidad_col: 'Cantidad Entrada Almacén'})
                                            .sort_values(['Mes Entrada', 'Talla'])
                                        )
                                        
                                        if not datos_tabla_tema.empty:
                                            # Crear tabla pivot para mejor visualización
                                            tabla_pivot = datos_tabla_tema.pivot_table(
                                                index='Mes Entrada',
                                                columns='Talla',
                                                values='Cantidad Entrada Almacén',
                                                fill_value=0
                                            ).round(0)
                                            tallas_orden = sorted(tabla_pivot.columns, key=custom_sort_key)
                                            tabla_pivot = tabla_pivot[tallas_orden]
                                            st.dataframe(
                                                tabla_pivot.style.format("{:,.0f}"),
                                                use_container_width=True,
                                                hide_index=False
                                            )
                                            total_temp = tabla_pivot.sum().sum()
                                            st.write(f"**Total Entrada Almacén:** {total_temp:,.0f}")
                                        else:
                                            st.info(f"No hay datos para el tema {tema}")
                else:
                    st.info("No hay datos de entrada en almacén disponibles para la familia seleccionada.")

            # --- Tabla de Pendientes de Entrega ---
            if not df_pendientes.empty:
                st.markdown("---")
                viz_title("Pendientes de Entrega")
                
                # Buscar la columna correcta para cantidad
                cantidad_col_pendientes = None
                for col in df_pendientes.columns:
                    if 'cantidad' in col.lower() and 'pedida' not in col.lower():
                        cantidad_col_pendientes = col
                        break
                
                if cantidad_col_pendientes is None:
                    # Si no encontramos una columna específica, usar 'Cantidad' o la primera columna numérica
                    numeric_cols = df_pendientes.select_dtypes(include=[np.number]).columns
                    if 'Cantidad' in df_pendientes.columns:
                        cantidad_col_pendientes = 'Cantidad'
                    elif len(numeric_cols) > 0:
                        cantidad_col_pendientes = numeric_cols[0]
                    else:
                        st.error("No se encontró una columna de cantidad válida para pendientes")
                        cantidad_col_pendientes = 'Cantidad'  # Fallback
                
                # Preparar datos de pendientes por talla
                datos_pendientes = (
                    df_pendientes.groupby(['Talla'])[cantidad_col_pendientes]
                    .sum()
                    .reset_index()
                    .rename(columns={cantidad_col_pendientes: 'Cantidad Pendiente'})
                    .sort_values('Talla', key=lambda x: x.map(custom_sort_key))
                )
                
                if not datos_pendientes.empty:
                    # Mostrar tabla de pendientes
                    st.dataframe(
                        datos_pendientes.style.format({
                            'Cantidad Pendiente': '{:,.0f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Mostrar total
                    total_pendientes = datos_pendientes['Cantidad Pendiente'].sum()
                    st.write(f"**Total Pendientes de Entrega:** {total_pendientes:,.0f}")
                    
                    # Mostrar información adicional
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total productos pendientes", len(df_pendientes))
                    with col2:
                        st.metric("Tallas diferentes", len(datos_pendientes))
                    with col3:
                        st.metric("Promedio por talla", f"{total_pendientes/len(datos_pendientes):,.0f}")
                else:
                    st.info("No hay datos de pendientes de entrega para mostrar.")
            else:
                st.info("No hay productos pendientes de entrega.")

            # --- Tabla de Productos Sin Familia Asignada ---
            if not df_sin_familia.empty:
                st.markdown("---")
                viz_title("Productos Sin Familia Asignada")
                
                # Buscar la columna correcta para cantidad
                cantidad_col_sin_familia = None
                for col in df_sin_familia.columns:
                    if 'cantidad' in col.lower() and 'pedida' not in col.lower():
                        cantidad_col_sin_familia = col
                        break
                
                if cantidad_col_sin_familia is None:
                    # Si no encontramos una columna específica, usar 'Cantidad' o la primera columna numérica
                    numeric_cols = df_sin_familia.select_dtypes(include=[np.number]).columns
                    if 'Cantidad' in df_sin_familia.columns:
                        cantidad_col_sin_familia = 'Cantidad'
                    elif len(numeric_cols) > 0:
                        cantidad_col_sin_familia = numeric_cols[0]
                    else:
                        st.error("No se encontró una columna de cantidad válida para productos sin familia")
                        cantidad_col_sin_familia = 'Cantidad'  # Fallback
                
                # Verificar si existe la columna 'Modelo Artículo'
                if 'Modelo Artículo' in df_sin_familia.columns:
                    # Preparar datos de productos sin familia por Modelo Artículo
                    datos_sin_familia = (
                        df_sin_familia.groupby(['Modelo Artículo'])[cantidad_col_sin_familia]
                        .sum()
                        .reset_index()
                        .rename(columns={cantidad_col_sin_familia: 'Cantidad Total'})
                        .sort_values('Cantidad Total', ascending=False)
                    )
                    
                    if not datos_sin_familia.empty:
                        # Mostrar tabla de productos sin familia
                        st.dataframe(
                            datos_sin_familia.style.format({
                                'Cantidad Total': '{:,.0f}'
                            }),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Mostrar total
                        total_sin_familia = datos_sin_familia['Cantidad Total'].sum()
                        st.write(f"**Total Productos Sin Familia:** {total_sin_familia:,.0f}")
                        
                        # Mostrar información adicional
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total productos sin familia", len(df_sin_familia))
                        with col2:
                            st.metric("Modelos diferentes", len(datos_sin_familia))
                        with col3:
                            st.metric("Promedio por modelo", f"{total_sin_familia/len(datos_sin_familia):,.0f}")
                    else:
                        st.info("No hay datos de productos sin familia para mostrar.")
                else:
                    st.warning("No se encontró la columna 'Modelo Artículo' en los datos de productos sin familia.")
            else:
                st.info("No hay productos sin familia asignada.")

            # --- Tabla de Cantidad Pedida por Mes y Talla ---
            # Solo mostrar esta tabla cuando NO se han seleccionado tiendas específicas
            if not tiendas_especificas:
                st.markdown("---")
                viz_title("Cantidad Pedida por Mes y Talla")
                
                # Use robust column finding function for 'Cantidad Pedida'
                cantidad_pedida_col = get_cantidad_pedida_column(df_almacen_fam)
                
                if not df_almacen_fam.empty and cantidad_pedida_col is not None:
                    # Preparar datos de cantidad pedida
                    datos_pedida = (
                        df_almacen_fam.groupby(['Mes Entrada', 'Talla'])[cantidad_pedida_col]
                        .sum()
                        .reset_index()
                        .rename(columns={'Mes Entrada': 'Mes', cantidad_pedida_col: 'Cantidad Pedida'})
                        .sort_values(['Mes', 'Talla'])
                    )
                    
                    # Filtrar datos hasta el último mes de ventas
                    datos_pedida = datos_pedida[datos_pedida['Mes'] <= ultimo_mes_ventas]
                    
                    if not datos_pedida.empty:
                        # Crear tabla pivot para mejor visualización
                        tabla_pedida_pivot = datos_pedida.pivot_table(
                            index='Mes',
                            columns='Talla',
                            values='Cantidad Pedida',
                            fill_value=0
                        ).round(0)
                        
                        # Ordenar tallas usando la función custom_sort_key
                        tallas_orden = sorted(tabla_pedida_pivot.columns, key=custom_sort_key)
                        tabla_pedida_pivot = tabla_pedida_pivot[tallas_orden]
                        
                        # Mostrar la tabla
                        st.dataframe(
                            tabla_pedida_pivot.style.format("{:,.0f}"),
                            use_container_width=True,
                            hide_index=False
                        )
                        
                        # Mostrar total
                        total_pedida = tabla_pedida_pivot.sum().sum()
                        st.write(f"**Total Cantidad Pedida:** {total_pedida:,.0f}")
                    else:
                        st.info("No hay datos de cantidad pedida para la familia seleccionada.")
                else:
                    st.info("No hay datos de cantidad pedida disponibles para la familia seleccionada.")

            # --- Ventas vs Traspasos por Tienda ---
            st.markdown("---")
            viz_title("Ventas vs Traspasos por Tienda")
            
            # Preparar datos de traspasos hasta la fecha máxima de ventas
            ultimo_mes_ventas = df_ventas['Mes'].max()
            df_traspasos_filtrado = df_traspasos_filtrado.copy()
            
            # Use robust column finding function
            fecha_enviado_col_vs = get_fecha_enviado_column(df_traspasos_filtrado)
            
            if fecha_enviado_col_vs is not None:
                # Convertir fecha de traspasos y filtrar hasta el último mes de ventas
                df_traspasos_filtrado['Mes Enviado'] = pd.to_datetime(df_traspasos_filtrado[fecha_enviado_col_vs]).dt.to_period('M').astype(str)
                df_traspasos_filtrado = df_traspasos_filtrado[df_traspasos_filtrado['Mes Enviado'] <= ultimo_mes_ventas]
            else:
                st.warning("⚠️ No se encontró la columna 'Fecha Enviado' en los datos de traspasos para la comparación con ventas.")
                # Skip this section if the column is not found
                st.info("No se pueden mostrar las comparaciones de ventas vs traspasos sin la columna 'Fecha Enviado'.")
                return
            
            # Agrupar ventas por tienda y temporada
            ventas_por_tienda_temp = df_ventas.groupby(['Tienda', 'Temporada'])['Cantidad'].sum().reset_index()
            ventas_por_tienda_temp['Tipo'] = 'Ventas'
            ventas_por_tienda_temp = ventas_por_tienda_temp.rename(columns={'Cantidad': 'Cantidad Total'})
            
            # Obtener ACTs que existen en ventas (limpiar espacios)
            act_en_ventas = df_ventas['ACT'].str.strip().unique()
            
            # Limpiar ACTs en traspasos también
            df_traspasos_filtrado['ACT_clean'] = df_traspasos_filtrado['ACT'].str.strip()
            
            # Filtrar traspasos para solo incluir ACTs que están en ventas
            df_traspasos_filtrado_act = df_traspasos_filtrado[df_traspasos_filtrado['ACT_clean'].isin(act_en_ventas)]
            
            # Agrupar traspasos por tienda y temporada
            if not df_traspasos_filtrado_act.empty:
                # Asegurar que la columna Temporada existe en traspasos
                if 'Temporada' not in df_traspasos_filtrado_act.columns:
                    temporada_columns = [col for col in df_traspasos_filtrado_act.columns if 'temporada' in col.lower() or 'season' in col.lower()]
                    if temporada_columns:
                        df_traspasos_filtrado_act['Temporada'] = df_traspasos_filtrado_act[temporada_columns[0]]
                    else:
                        df_traspasos_filtrado_act['Temporada'] = 'Sin Temporada'
                else:
                    df_traspasos_filtrado_act['Temporada'] = df_traspasos_filtrado_act['Temporada'].fillna('Sin Temporada')
                
                # Limpiar temporada en traspasos para que coincida con ventas
                df_traspasos_filtrado_act['Temporada'] = df_traspasos_filtrado_act['Temporada'].str.strip().str[:5]
                
                traspasos_por_tienda_temp = df_traspasos_filtrado_act.groupby(['Tienda', 'Temporada'])['Enviado'].sum().reset_index()
                traspasos_por_tienda_temp['Tipo'] = 'Traspasos'
                traspasos_por_tienda_temp = traspasos_por_tienda_temp.rename(columns={'Enviado': 'Cantidad Total'})
            else:
                # Si no hay traspasos filtrados, crear un DataFrame vacío con la estructura correcta
                traspasos_por_tienda_temp = pd.DataFrame(columns=['Tienda', 'Temporada', 'Cantidad Total', 'Tipo'])
            
            # Combinar datos
            datos_comparacion = pd.concat([ventas_por_tienda_temp, traspasos_por_tienda_temp], ignore_index=True)
            
            if not datos_comparacion.empty:
                # Obtener top 20 tiendas por ventas totales
                top_tiendas_ventas = df_ventas.groupby('Tienda')['Cantidad'].sum().nlargest(20).index.tolist()
                
                # Filtrar datos para top 20 tiendas
                datos_top_tiendas = datos_comparacion[datos_comparacion['Tienda'].isin(top_tiendas_ventas)]
                
                if not datos_top_tiendas.empty:
                    # Crear gráfico con exactamente 2 barras por tienda (Ventas y Traspasos)
                    # Preparar datos para el nuevo formato
                    ventas_data = datos_top_tiendas[datos_top_tiendas['Tipo'] == 'Ventas'].copy()
                    traspasos_data = datos_top_tiendas[datos_top_tiendas['Tipo'] == 'Traspasos'].copy()
                    
                    # Obtener colores de temporada
                    temporada_colors = get_temporada_colors(df_ventas)
                    
                    # Crear figura
                    fig = go.Figure()
                    
                    # Obtener todas las tiendas únicas
                    tiendas_unicas = sorted(datos_top_tiendas['Tienda'].unique())
                    temporadas = sorted(datos_top_tiendas['Temporada'].unique())
                    
                    # Definir diferentes tonos de amarillo para traspasos por temporada
                    yellow_colors = ['#ffff00', '#ffeb3b', '#ffc107', '#ff9800', '#ff5722', '#f57c00', '#ef6c00', '#e65100']
                    
                    # Crear datos para cada tienda con dos barras (Ventas y Traspasos)
                    for tienda in tiendas_unicas:
                        # Datos de ventas para esta tienda
                        ventas_tienda = ventas_data[ventas_data['Tienda'] == tienda]
                        traspasos_tienda = traspasos_data[traspasos_data['Tienda'] == tienda]
                        
                        # Agregar barra de VENTAS (dividida por temporada)
                        if not ventas_tienda.empty:
                            for i, temporada in enumerate(temporadas):
                                ventas_temp = ventas_tienda[ventas_tienda['Temporada'] == temporada]
                                if not ventas_temp.empty:
                                    fig.add_trace(go.Bar(
                                        name=f'Ventas - {temporada}',
                                        x=[f'{tienda} - Ventas'],
                                        y=ventas_temp['Cantidad Total'],
                                        marker_color=temporada_colors.get(temporada, '#1f77b4'),
                                        text=ventas_temp['Cantidad Total'],
                                        texttemplate='%{text:,.0f}',
                                        textposition='inside',
                                        hovertemplate=f"Tienda: {tienda}<br>Tipo: Ventas<br>Temporada: {temporada}<br>Cantidad: %{{y:,.0f}}<extra></extra>",
                                        opacity=0.8,
                                        showlegend=True if tienda == tiendas_unicas[0] else False,  # Solo mostrar legend para la primera tienda
                                        legendgroup=f'Ventas - {temporada}'
                                    ))
                        
                        # Agregar barra de TRASPASOS (dividida por temporada)
                        if not traspasos_tienda.empty:
                            for i, temporada in enumerate(temporadas):
                                traspasos_temp = traspasos_tienda[traspasos_tienda['Temporada'] == temporada]
                                if not traspasos_temp.empty:
                                    # Usar diferentes tonos de amarillo para cada temporada
                                    yellow_color = yellow_colors[i % len(yellow_colors)]
                                    fig.add_trace(go.Bar(
                                        name=f'Traspasos - {temporada}',
                                        x=[f'{tienda} - Traspasos'],
                                        y=traspasos_temp['Cantidad Total'],
                                        marker_color=yellow_color,  # Diferentes tonos de amarillo por temporada
                                        text=traspasos_temp['Cantidad Total'],
                                        texttemplate='%{text:,.0f}',
                                        textposition='inside',
                                        hovertemplate=f"Tienda: {tienda}<br>Tipo: Traspasos<br>Temporada: {temporada}<br>Cantidad: %{{y:,.0f}}<extra></extra>",
                                        opacity=0.8,
                                        showlegend=True if tienda == tiendas_unicas[0] else False,  # Solo mostrar legend para la primera tienda
                                        legendgroup=f'Traspasos - {temporada}'
                                    ))
                    
                    # Configurar layout
                    fig.update_layout(
                        title="Ventas vs Traspasos por Tienda",
                        xaxis_title="Tienda",
                        yaxis_title="Cantidad Total",
                        barmode='stack',  # Barras apiladas por temporada
                        xaxis_tickangle=45,
                        showlegend=True,
                        margin=dict(t=30, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar tabla resumen con breakdown por temporada
                    st.subheader("Resumen de Ventas vs Traspasos por Temporada")
                    
                    # Tabla con breakdown por temporada
                    resumen_temporada = datos_top_tiendas.groupby(['Tienda', 'Tipo', 'Temporada'])['Cantidad Total'].sum().reset_index()
                    resumen_pivot_temp = resumen_temporada.pivot_table(
                        index=['Tienda', 'Temporada'], 
                        columns='Tipo', 
                        values='Cantidad Total', 
                        fill_value=0
                    ).reset_index()
                    
                    # Calcular totales por tienda
                    resumen_totales = datos_top_tiendas.groupby(['Tienda', 'Tipo'])['Cantidad Total'].sum().reset_index()
                    resumen_pivot_totales = resumen_totales.pivot(index='Tienda', columns='Tipo', values='Cantidad Total').fillna(0)
                    resumen_pivot_totales['Diferencia'] = resumen_pivot_totales['Ventas'] - resumen_pivot_totales['Traspasos']
                    
                    
                    resumen_pivot_totales['Eficiencia %'] = (resumen_pivot_totales['Ventas'] / resumen_pivot_totales['Traspasos'] * 100).fillna(0)

                    # Calcular Devoluciones (cantidad negativa) por tienda
                    devoluciones_por_tienda = df_ventas[df_ventas['Cantidad'] < 0].groupby('Tienda')['Cantidad'].sum().abs()
                    resumen_pivot_totales['Devoluciones'] = devoluciones_por_tienda.reindex(resumen_pivot_totales.index).fillna(0)
                    
                    # Calcular Ratio de devolución (Devoluciones / Ventas * 100)
                    resumen_pivot_totales['Ratio de devolución %'] = (resumen_pivot_totales['Devoluciones'] / resumen_pivot_totales['Ventas'] * 100).fillna(0)
                    
                    resumen_pivot_totales = resumen_pivot_totales.round(2)
                    
                    # Mostrar tabla de totales
                    st.write("**Totales por Tienda:**")
                    st.dataframe(
                        resumen_pivot_totales.style.format({
                            'Ventas': '{:,.0f}',
                            'Traspasos': '{:,.0f}',
                            'Diferencia': '{:,.0f}',
                            'Devoluciones': '{:,.0f}',
                            'Eficiencia %': '{:.1f}%',
                            'Ratio de devolución %': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
                    
                    # Mostrar tabla detallada por temporada
                    st.write("**Detalle por Temporada:**")
                    st.dataframe(
                        resumen_pivot_temp.style.format({
                            'Ventas': '{:,.0f}',
                            'Traspasos': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No hay datos suficientes para mostrar la comparación.")
            else:
                st.info("No hay datos de traspasos disponibles para la comparación.")

            
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Top tiendas con menos ventas por unidades
                viz_title("Top tiendas con menos ventas")
                ventas_por_tienda = df_ventas.groupby('NombreTPV').agg({
                    'Cantidad': 'sum',
                    'Ventas Dinero': 'sum'
                }).reset_index()
                ventas_por_tienda.columns = ['Tienda', 'Unidades Vendidas', 'Ventas (€)']
                bottom_30_tiendas = ventas_por_tienda.nsmallest(30, 'Unidades Vendidas')
                
                fig = px.bar(
                    bottom_30_tiendas,
                    x='Tienda',
                    y='Unidades Vendidas',
                    color='Unidades Vendidas',
                    color_continuous_scale=COLOR_GRADIENT,
                    height=400
                )
                fig.update_layout(
                    xaxis_tickangle=45,
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(
                    texttemplate='%{y:,.0f} uds',
                    textposition='outside',
                    hovertemplate="Tienda: %{x}<br>Unidades: %{y:,}<br>Ventas: %{customdata:,.2f}€<extra></extra>",
                    customdata=bottom_30_tiendas['Ventas (€)'],
                    opacity=0.8
                )
                st.plotly_chart(fig, use_container_width=True)

            # Nueva fila: Top tiendas con más ventas por ventas (€)
            col3, col4 = st.columns(2)
            with col3:
                viz_title("Top tiendas con más ventas")
                top_30_tiendas = ventas_por_tienda.nlargest(30, 'Ventas (€)')
                fig = px.bar(
                    top_30_tiendas,
                    x='Tienda',
                    y='Ventas (€)',
                    color='Ventas (€)',
                    color_continuous_scale=COLOR_GRADIENT,
                    height=400,
                    labels={'Tienda': 'Tienda', 'Ventas (€)': 'Ventas (€)', 'Unidades Vendidas': 'Unidades'}
                )
                fig.update_layout(
                    xaxis_tickangle=45,
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(
                    texttemplate='%{y:,.2f}€',
                    textposition='outside',
                    hovertemplate="Tienda: %{x}<br>Ventas: %{y:,.2f}€<br>Unidades: %{customdata:,}<extra></extra>",
                    customdata=top_30_tiendas['Unidades Vendidas'],
                    opacity=0.8
                )
                st.plotly_chart(fig, use_container_width=True)
            with col4:
                st.empty()

            # Par 2: Unidades por Talla y Ventas por Familia
            col5, col6 = st.columns(2)
            
            with col5:
                viz_title("Unidades Vendidas por Talla")
                familias = sorted(df_ventas['Familia'].unique())
                familia_seleccionada = st.selectbox("Selecciona una familia:", familias)
                
                df_familia = df_ventas[df_ventas['Familia'] == familia_seleccionada]
                
                # 1. Obtener todas las tallas únicas para la familia seleccionada
                tallas_presentes = df_familia['Talla'].dropna().unique()
                
                # 2. Ordenarlas con la lógica personalizada
                tallas_orden = sorted(tallas_presentes, key=custom_sort_key)
                
                # 3. Agrupar y sumar las cantidades
                tallas_sumadas = df_familia.groupby('Talla')['Cantidad'].sum()
                
                # 4. Reindexar para asegurar el orden y la inclusión de todas las tallas (incluso con suma 0)
                tallas_grafico = tallas_sumadas.reindex(tallas_orden, fill_value=0).reset_index()

                # Mostrar gráfica de barras por talla
                fig = px.bar(
                    tallas_grafico,
                    x='Talla',
                    y='Cantidad',
                    text='Cantidad',
                    color='Cantidad',
                    color_continuous_scale=COLOR_GRADIENT,
                    height=400
                )
                fig.update_layout(
                    xaxis_title="Talla",
                    yaxis_title="Unidades Vendidas",
                    showlegend=False,
                    xaxis={'categoryorder': 'array', 'categoryarray': tallas_orden},
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(
                    texttemplate='%{text:,.0f}', 
                    textposition='outside',
                    opacity=0.8
                )
                st.plotly_chart(fig, use_container_width=True)

                # Mostrar tabla de tallas (letras y números)
                st.dataframe(tallas_grafico, use_container_width=True)

            with col6:
                # Ventas por Familia: tooltip mejorado
                viz_title("Ventas por Familia")
                ventas_familia = df_ventas.groupby('Familia').agg({
                    'Cantidad': 'sum',
                    'Ventas Dinero': 'sum'
                }).reset_index()
                ventas_familia = ventas_familia.sort_values('Ventas Dinero', ascending=True)

                fig = px.bar(
                    ventas_familia,
                    x='Ventas Dinero',
                    y='Familia',
                    orientation='h',
                    color='Ventas Dinero',
                    color_continuous_scale=COLOR_GRADIENT,
                    height=400,
                    labels={'Ventas Dinero': 'Ventas (€)', 'Cantidad': 'Unidades'}
                )
                fig.update_traces(
                    texttemplate='%{text:,.0f} uds',
                    textposition='outside',
                    hovertemplate="Familia: %{y}<br>Ventas: %{x:,.2f}€<br>Unidades: %{text:,.0f}<extra></extra>",
                    opacity=0.8
                )
                st.plotly_chart(fig, use_container_width=True)

            # --- REVISED: Top/Bottom 10 Complete Descriptions by Family ---
            st.markdown("---")
            viz_title("Análisis de Descripciones por Familia")

            desc_path = os.path.join('data', 'datos_descripciones.xlsx')
            if os.path.exists(desc_path):
                try:
                    df_desc = pd.read_excel(desc_path, engine='openpyxl')
                    
                    desc_cols = ['MANGA', 'CUELLO', 'TEJIDO', 'DETALLE', 'ESTILO', 'CORTE']
                    
                    if all(col in df_desc.columns for col in desc_cols):
                        # --- FILTROS ---
                        col_filter1, col_filter2 = st.columns(2)
                        
                        with col_filter1:
                             # Unimos con ventas primero para solo mostrar familias relevantes
                            ventas_desc = df_ventas.copy()
                            ventas_desc['act_clean'] = ventas_desc['ACT'].astype(str).str[:-1]
                            ventas_con_desc_pre = ventas_desc.merge(
                                df_desc[['ACT'] + desc_cols],
                                left_on='act_clean',
                                right_on='ACT',
                                how='inner'
                            )
                            familias_disponibles = sorted(ventas_con_desc_pre['Familia'].dropna().unique())
                            familia_seleccionada = st.selectbox(
                                "Selecciona una Familia:", 
                                familias_disponibles, 
                                key="familia_desc_selector"
                            )

                        with col_filter2:
                            opciones_desc = ["Descripción Completa"] + desc_cols
                            tipo_descripcion = st.selectbox(
                                "Selecciona Tipo de Descripción:", 
                                opciones_desc, 
                                key="tipo_desc_selector"
                            )

                        # --- LÓGICA DE DATOS ---
                        if tipo_descripcion == "Descripción Completa":
                            df_desc['Descripción Analizada'] = df_desc[desc_cols].fillna('').apply(
                                lambda row: ' - '.join(row.values.astype(str)).strip(), axis=1
                            )
                        else:
                            df_desc['Descripción Analizada'] = df_desc[tipo_descripcion].fillna('N/A')
                        
                        df_desc_clean = df_desc[['ACT', 'Descripción Analizada']].copy().dropna()
                        
                        ventas_con_desc = ventas_desc.merge(
                            df_desc_clean,
                            left_on='act_clean',
                            right_on='ACT',
                            how='inner'
                        )
                        
                        df_familia_desc = ventas_con_desc[ventas_con_desc['Familia'] == familia_seleccionada]
                        
                        desc_group = df_familia_desc.groupby('Descripción Analizada').agg({
                            'Ventas Dinero': 'sum',
                            'Cantidad': 'sum'
                        }).reset_index()
                        
                        # Excluir descripciones vacías o sin valor real
                        if tipo_descripcion == "Descripción Completa":
                            desc_group = desc_group[desc_group['Descripción Analizada'].str.replace(' - ', '').str.strip() != '']
                        else:
                            desc_group = desc_group[desc_group['Descripción Analizada'] != 'N/A']

                        if not desc_group.empty:
                            desc_group = desc_group.sort_values('Ventas Dinero', ascending=False)
                            top10 = desc_group.head(10)
                            bottom10 = desc_group.tail(10)

                            colA, colB = st.columns(2)
                            with colA:
                                viz_title(f'Top 10 en {tipo_descripcion}')
                                fig = px.bar(
                                    top10, 
                                    x='Ventas Dinero', 
                                    y='Descripción Analizada', 
                                    orientation='h', 
                                    color='Ventas Dinero', 
                                    color_continuous_scale=COLOR_GRADIENT,
                                    text='Cantidad'
                                )
                                fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending', 'title': ''}, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                                fig.update_traces(texttemplate='%{text:,.0f} uds', textposition='outside', hovertemplate="Descripción: %{y}<br>Ventas: %{x:,.2f}€<br>Unidades: %{text:,.0f}<extra></extra>", opacity=0.8)
                        
                            with colB:
                                viz_title(f'Bottom 10 en {tipo_descripcion}')
                                fig = px.bar(
                                    bottom10, 
                                    x='Ventas Dinero', 
                                    y='Descripción Analizada', 
                                    orientation='h', 
                                    color='Ventas Dinero', 
                                    color_continuous_scale=COLOR_GRADIENT,
                                    text='Cantidad'
                                )
                                fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending', 'title': ''}, margin=dict(t=30, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                                fig.update_traces(texttemplate='%{text:,.0f} uds', textposition='outside', hovertemplate="Descripción: %{y}<br>Ventas: %{x:,.2f}€<br>Unidades: %{text:,.0f}<extra></extra>", opacity=0.8)
                                st.plotly_chart(fig, use_container_width=True, key=f"bottom10_{tipo_descripcion}_{familia_seleccionada}")
                        else:
                            st.info(f"No hay datos de '{tipo_descripcion}' para la familia '{familia_seleccionada}'.")
                    else:
                        st.warning(f"Una o más columnas de descripción no se encontraron. Se necesitan: {desc_cols}")
                except Exception as e:
                    st.error(f"Error crítico al procesar las descripciones de productos: {e}")
            else:
                st.warning("Archivo `datos_descripciones.xlsx` no encontrado en la carpeta `data/`.")
            # --- END REVISED ---

        except Exception as e:
            st.error(f"Error al calcular KPIs: {e}")

    elif seccion == "Geográfico y Tiendas":
        # Preparar datos
        ventas_por_zona = df_ventas.groupby('Zona geográfica')['Cantidad'].sum().reset_index()
        ventas_por_tienda = df_ventas.groupby('NombreTPV')['Cantidad'].sum().reset_index()
        ventas_por_fam_zona = df_ventas.groupby(['Zona geográfica', 'Familia'])['Cantidad'].sum().reset_index()
        ventas_tienda_mes = df_ventas.groupby(['Mes', 'NombreTPV'])['Cantidad'].sum().reset_index()
        tiendas_por_zona = df_ventas[['NombreTPV', 'Zona geográfica']].drop_duplicates().groupby('Zona geográfica').count().reset_index()

        # Primera fila de visualizaciones
        col1, col2 = st.columns(2)
        
        with col1:
            # Ventas por Zona
            fig = px.bar(ventas_por_zona, 
                        x='Zona geográfica', 
                        y='Cantidad',
                        color='Cantidad',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='Cantidad')
            fig.update_layout(
                title="Ventas por Zona",
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top 5 Tiendas
            top_5_tiendas = ventas_por_tienda.nlargest(5, 'Cantidad')
            fig = px.bar(top_5_tiendas, 
                        x='NombreTPV', 
                        y='Cantidad',
                        color='Cantidad',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='Cantidad')
            fig.update_layout(
                title="Top 5 Tiendas",
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        # Segunda fila de visualizaciones
        col3, col4 = st.columns(2)
        
        with col3:
            # Tiendas por Zona
            fig = px.bar(tiendas_por_zona, 
                        x='Zona geográfica', 
                        y='NombreTPV',
                        color='NombreTPV',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='NombreTPV')
            fig.update_layout(
                title="Tiendas por Zona",
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Distribución Mensual de Ventas
            pivot_tienda_mes = ventas_tienda_mes.pivot(index='Mes', columns='NombreTPV', values='Cantidad').fillna(0)
            fig = px.line(pivot_tienda_mes, 
                         color_discrete_sequence=COLOR_GRADIENT)
            fig.update_layout(
                title="Distribución Mensual de Ventas",
                showlegend=True,
                legend_title_text='Tienda',
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        # Tercera fila de visualizaciones
        col5, col6 = st.columns(2)
        
        with col5:
            # Evolución Mensual por Zona
            zona_mes_evol = df_ventas.groupby(['Mes', 'Zona geográfica'])['Cantidad'].sum().reset_index()
            fig = px.line(zona_mes_evol, 
                         x='Mes', 
                         y='Cantidad',
                         color='Zona geográfica',
                         color_discrete_sequence=COLOR_GRADIENT)
            fig.update_layout(
                title="Evolución Mensual por Zona",
                showlegend=True,
                legend_title_text='Zona Geográfica',
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            # Mapa de Ventas - España e Italia
            viz_title("Mapa de Ventas - España e Italia")
            
            # Separar datos por país
            df_espana = df_ventas[~df_ventas['NombreTPV'].isin(TIENDAS_EXTRANJERAS)].copy()
            df_italia = df_ventas[df_ventas['NombreTPV'].isin(TIENDAS_EXTRANJERAS)].copy()
            
            # Procesar datos de España
            df_espana['Ciudad'] = df_espana['NombreTPV'].str.extract(r'ET\d{1,2}-([\w\s\.\(\)]+)')[0]
            df_espana['Ciudad'] = (
                df_espana['Ciudad']
                .str.upper()
                .str.replace(r'ECITRUCCO|ECI|XANADU|TRUCCO|CORT.*', '', regex=True)
                .str.strip()
            )

            coordenadas_espana = {
                'MADRID': (40.4168, -3.7038),
                'SEVILLA': (37.3886, -5.9823),
                'MALAGA': (36.7213, -4.4214),
                'VALENCIA': (39.4699, -0.3763),
                'VIGO': (42.2406, -8.7207),
                'MURCIA': (37.9834, -1.1299),
                'SALAMANCA': (40.9701, -5.6635),
                'CORDOBA': (37.8882, -4.7794),
                'BILBAO': (43.2630, -2.9350),
                'ZARAGOZA': (41.6488, -0.8891),
                'JAEN': (37.7796, -3.7849),
                'GIJON': (43.5453, -5.6615),
                'ALBACETE': (38.9943, -1.8585),
                'GRANADA': (37.1773, -3.5986),
                'CARTAGENA': (37.6051, -0.9862),
                'TARRAGONA': (41.1189, 1.2445),
                'LEON': (42.5987, -5.5671),
                'SANTANDER': (43.4623, -3.8099),
                'PAMPLONA': (42.8125, -1.6458),
                'VITORIA': (42.8467, -2.6727),
                'CASTELLON': (39.9864, -0.0513),
                'CADIZ': (36.5271, -6.2886),
                'JEREZ': (36.6850, -6.1261),
                'AVILES': (43.5560, -5.9222),
                'BADAJOZ': (38.8794, -6.9707)
            }

            # Procesar datos de Italia
            df_italia['Ciudad'] = df_italia['NombreTPV'].str.extract(r'I\d{3}COIN([A-Z]+)')[0]
            df_italia['Ciudad'] = df_italia['Ciudad'].fillna('MILANO')  # Default para tiendas sin ciudad extraída

            coordenadas_italia = {
                'BERGAMO': (45.6983, 9.6773),
                'VARESE': (45.8206, 8.8256),
                'BARICASAMASSIMA': (40.9634, 16.7514),
                'MILANO5GIORNATE': (45.4642, 9.1900),
                'ROMACINECITTA': (41.9028, 12.4964),
                'GENOVA': (44.4056, 8.9463),
                'SASSARI': (40.7259, 8.5557),
                'CATANIA': (37.5079, 15.0830),
                'CAGLIARI': (39.2238, 9.1217),
                'LECCE': (40.3519, 18.1720),
                'MILANOCANTORE': (45.4642, 9.1900),
                'MESTRE': (45.4903, 12.2424),
                'PADOVA': (45.4064, 11.8768),
                'FIRENZE': (43.7696, 11.2558),
                'ROMASANGIOVANNI': (41.9028, 12.4964),
                'MILANO': (45.4642, 9.1900)
            }

            # Crear mapas separados
            col_map1, col_map2 = st.columns(2)
            
            with col_map1:
                # Mapa de España
                df_espana['lat'] = df_espana['Ciudad'].map(lambda c: coordenadas_espana.get(c, (None, None))[0])
                df_espana['lon'] = df_espana['Ciudad'].map(lambda c: coordenadas_espana.get(c, (None, None))[1])
                df_espana = df_espana.dropna(subset=['lat', 'lon'])

                ventas_ciudad_espana = df_espana.groupby(['Ciudad', 'lat', 'lon'])['Cantidad'].sum().reset_index()

                if not ventas_ciudad_espana.empty:
                    # Normalizar tamaños para España
                    min_size = 10
                    max_size = 50
                    ventas_ciudad_espana['marker_size'] = min_size + (ventas_ciudad_espana['Cantidad'] - ventas_ciudad_espana['Cantidad'].min()) / \
                        (ventas_ciudad_espana['Cantidad'].max() - ventas_ciudad_espana['Cantidad'].min()) * (max_size - min_size)

                    fig_espana = px.scatter_mapbox(
                        ventas_ciudad_espana,
                    lat='lat',
                    lon='lon',
                        size='marker_size',
                    color='Cantidad',
                    hover_name='Ciudad',
                        hover_data={
                            'marker_size': False,
                            'Cantidad': True
                        },
                        color_continuous_scale=COLOR_GRADIENT,
                        zoom=5,
                        height=300,
                        title="España"
                    )
                    fig_espana.update_layout(
                        mapbox_style='carto-positron',
                        margin=dict(t=30, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_espana, use_container_width=True)
                else:
                    st.info("No hay datos disponibles para España.")

            with col_map2:
                # Mapa de Italia
                df_italia['lat'] = df_italia['Ciudad'].map(lambda c: coordenadas_italia.get(c, (None, None))[0])
                df_italia['lon'] = df_italia['Ciudad'].map(lambda c: coordenadas_italia.get(c, (None, None))[1])
                df_italia = df_italia.dropna(subset=['lat', 'lon'])

                ventas_ciudad_italia = df_italia.groupby(['Ciudad', 'lat', 'lon'])['Cantidad'].sum().reset_index()

                if not ventas_ciudad_italia.empty:
                    # Normalizar tamaños para Italia
                    min_size = 10
                    max_size = 50
                    ventas_ciudad_italia['marker_size'] = min_size + (ventas_ciudad_italia['Cantidad'] - ventas_ciudad_italia['Cantidad'].min()) / \
                        (ventas_ciudad_italia['Cantidad'].max() - ventas_ciudad_italia['Cantidad'].min()) * (max_size - min_size)

                    fig_italia = px.scatter_mapbox(
                        ventas_ciudad_italia,
                        lat='lat',
                        lon='lon',
                        size='marker_size',
                        color='Cantidad',
                        hover_name='Ciudad',
                        hover_data={
                            'marker_size': False,
                            'Cantidad': True
                        },
                        color_continuous_scale=COLOR_GRADIENT,
                        zoom=5,
                        height=300,
                        title="Italia"
                    )
                    fig_italia.update_layout(
                        mapbox_style='carto-positron',
                        margin=dict(t=30, b=0, l=0, r=0),
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_italia, use_container_width=True)
                else:
                    st.info("No hay datos disponibles para Italia.")
    
    elif seccion == "Stock y Traspasos":
        # Preparación de datos
        df_traspasos['Fecha Documento'] = pd.to_datetime(df_traspasos['Fecha Documento'], errors='coerce')
        df_traspasos['Mes'] = df_traspasos['Fecha Documento'].dt.to_period('M').astype(str)
        df_traspasos['Tienda'] = df_traspasos['NombreTpvDestino'].astype(str)

        # Primera fila de visualizaciones
        col1, col2 = st.columns(2)
        
        with col1:
            # Tiendas con más Traspasos
            top_tiendas = df_traspasos.groupby('Tienda')['Enviado'].sum().nlargest(10).reset_index()
            fig = px.bar(top_tiendas,
                        x='Tienda',
                        y='Enviado',
                        color='Enviado',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='Enviado')
            fig.update_layout(
                title="Tiendas con más Traspasos",
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Tiendas con menos Traspasos
            bottom_tiendas = df_traspasos.groupby('Tienda')['Enviado'].sum().nsmallest(10).reset_index()
            fig = px.bar(bottom_tiendas,
                        x='Tienda',
                        y='Enviado',
                        color='Enviado',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='Enviado')
            fig.update_layout(
                title="Tiendas con menos Traspasos",
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        # Segunda fila de visualizaciones
        col3, col4 = st.columns(2)
        
        with col3:
            # Traspasos por Mes
            traspasos_mes = df_traspasos.groupby('Mes')['Enviado'].sum().reset_index()
            fig = px.line(traspasos_mes,
                         x='Mes',
                         y='Enviado',
                         markers=True,
                         color_discrete_sequence=COLOR_GRADIENT)
            fig.update_layout(
                title="Traspasos por Mes",
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Traspasos por Zona
            if 'Zona geográfica' in df_traspasos.columns:
                traspasos_zona = df_traspasos.groupby('Zona geográfica')['Enviado'].sum().reset_index()
                fig = px.bar(traspasos_zona,
                            x='Zona geográfica',
                            y='Enviado',
                            color='Enviado',
                            color_continuous_scale=COLOR_GRADIENT,
                            text='Enviado')
                fig.update_layout(
                    title="Traspasos por Zona",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos de zona geográfica disponibles para traspasos.")

        # Tercera fila de visualizaciones
        col5, col6 = st.columns(2)
        
        with col5:
            # Modelos más Traspasados
            modelo_column = 'ACT' if 'ACT' in df_traspasos.columns else 'Modelo'
            if modelo_column in df_traspasos.columns:
                top_modelos = df_traspasos.groupby(modelo_column)['Enviado'].sum().nlargest(10).reset_index()
                fig = px.bar(top_modelos,
                            x=modelo_column,
                            y='Enviado',
                            color='Enviado',
                            color_continuous_scale=COLOR_GRADIENT,
                            text='Enviado')
                fig.update_layout(
                    title="Modelos más Traspasados",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se encontró la columna de modelos (ACT o Modelo) en los datos de traspasos. Por favor, verifica que los datos contengan esta información.")

        with col6:
            # Traspasos por Talla
            if 'Talla' in df_traspasos.columns:
                traspasos_talla = df_traspasos.groupby('Talla')['Enviado'].sum().reset_index()
                fig = px.bar(traspasos_talla,
                            x='Talla',
                            y='Enviado',
                            color='Enviado',
                            color_continuous_scale=COLOR_GRADIENT,
                            text='Enviado')
                fig.update_layout(
                    title="Traspasos por Talla",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(traspasos_talla, use_container_width=True)
            else:
                st.info("No hay datos de talla disponibles para traspasos.")

    elif seccion == "Producto, Campaña, Devoluciones y Rentabilidad":
        devoluciones = df_ventas[df_ventas['Cantidad'] < 0].copy()
        ventas = df_ventas[df_ventas['Cantidad'] > 0].copy()

        # Calcular descuento real basado en la diferencia entre PVP y precio real de venta
        if all(col in df_ventas.columns for col in ['P.V.P.', 'Subtotal', 'Cantidad']):
            df_ventas['Precio Real Unitario'] = df_ventas['Subtotal'] / df_ventas['Cantidad']
            df_ventas['Descuento Real %'] = ((df_ventas['P.V.P.'] - df_ventas['Precio Real Unitario']) / df_ventas['P.V.P.'] * 100).fillna(0)
            df_ventas['Descuento Real %'] = df_ventas['Descuento Real %'].clip(0, 100)  # Limitar entre 0 y 100%

        # Organizar visualizaciones en pares
        col1, col2 = st.columns(2)
        
        with col1:
            # Productos más Vendidos
            identifier = 'ACT' if 'ACT' in df_ventas.columns else 'Producto'
            if identifier in df_ventas.columns:
                top_productos = df_ventas.groupby(identifier)['Cantidad'].sum().nlargest(10).reset_index()
                fig = px.bar(top_productos,
                            x=identifier,
                            y='Cantidad',
                            color='Cantidad',
                            color_continuous_scale=COLOR_GRADIENT,
                            text='Cantidad')
                fig.update_layout(
                    title="Productos más Vendidos",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se encontró la columna de productos en los datos.")

        with col2:
            # Devoluciones por Familia
            if 'Familia' in devoluciones.columns:
                dev_familia = devoluciones.groupby('Familia')['Cantidad'].sum().abs().nlargest(10).reset_index()
                fig = px.bar(dev_familia,
                            x='Familia',
                            y='Cantidad',
                            color='Cantidad',
                            color_continuous_scale=COLOR_GRADIENT,
                            text='Cantidad')
                fig.update_layout(
                    title="Devoluciones por Familia",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos de familia disponibles para devoluciones.")

        col3, col4 = st.columns(2)
        
        with col3:
            # Rentabilidad por Producto
            if all(col in df_ventas.columns for col in ['P.V.P.', 'Cantidad', 'Subtotal']):
                rentabilidad = df_ventas.groupby(identifier).agg({
                    'Subtotal': 'sum',
                    'Cantidad': 'sum'
                }).reset_index()
                rentabilidad = rentabilidad.nlargest(10, 'Subtotal')
                
                fig = px.bar(rentabilidad,
                            x=identifier,
                            y='Subtotal',
                            color='Subtotal',
                            color_continuous_scale=COLOR_GRADIENT,
                            text='Subtotal')
                fig.update_layout(
                    title="Rentabilidad por Producto",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(
                    texttemplate='%{text:,.0f}€',
                    opacity=0.8
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos suficientes para calcular la rentabilidad.")

        with col4:
            # Devoluciones por Color
            color_column = 'Descripción Color' if 'Descripción Color' in devoluciones.columns else 'Color'
            if color_column in devoluciones.columns:
                dev_color = devoluciones.groupby(color_column)['Cantidad'].sum().abs().nlargest(10).reset_index()
                fig = px.bar(dev_color,
                            x=color_column,
                            y='Cantidad',
                            color='Cantidad',
                            color_continuous_scale=COLOR_GRADIENT,
                            text='Cantidad')
                fig.update_layout(
                    title="Devoluciones por Color",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos de color disponibles para devoluciones.")

        col5, col6 = st.columns(2)
        
        with col5:
            # Devoluciones por Talla
            if 'Talla' in devoluciones.columns:
                dev_talla = devoluciones.groupby('Talla')['Cantidad'].sum().abs().reset_index()
                fig = px.bar(dev_talla,
                            x='Talla',
                            y='Cantidad',
                            color='Cantidad',
                            color_continuous_scale=COLOR_GRADIENT,
                            text='Cantidad')
                fig.update_layout(
                    title="Devoluciones por Talla",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(dev_talla, use_container_width=True)
            else:
                st.info("No hay datos de talla disponibles para devoluciones.")

        with col6:
            # Ventas por Descuento Real
            if 'Descuento Real %' in df_ventas.columns:
                # Crear rangos de descuento para mejor visualización
                df_ventas['Rango Descuento'] = pd.cut(
                    df_ventas['Descuento Real %'],
                    bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                           '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
                )
                
                ventas_desc = df_ventas.groupby('Rango Descuento').agg({
            'Cantidad': 'sum',
                    'Subtotal': 'sum'
        }).reset_index()

                fig = px.bar(ventas_desc,
                            x='Rango Descuento',
                            y='Cantidad',
                            color='Subtotal',
                            color_continuous_scale=COLOR_GRADIENT,
                            text='Cantidad')
                fig.update_layout(
                    title="Ventas por Rango de Descuento",
                    showlegend=False,
                    xaxis_tickangle=45,
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos suficientes para calcular los descuentos.")

    # Robust column name handling functions
    def find_column_by_pattern(df, pattern):
        """
        Find a column in DataFrame that matches a pattern, handling whitespace and encoding issues
        """
        if df is None or not isinstance(df, pd.DataFrame):
            return None
        
        pattern_clean = pattern.strip().lower()
        for col in df.columns:
            col_clean = str(col).strip().lower()
            if pattern_clean in col_clean or col_clean in pattern_clean:
                return col
        return None

    def get_fecha_enviado_column(df_traspasos):
        """
        Get the 'Fecha Enviado' column from traspasos DataFrame with robust handling
        """
        if df_traspasos is None or not isinstance(df_traspasos, pd.DataFrame):
            return None
        
        # Try multiple patterns to find the column
        patterns = ['Fecha Enviado', 'FechaEnviado', 'Fecha_Enviado', 'fecha enviado', 'fechaenviado']
        
        for pattern in patterns:
            col = find_column_by_pattern(df_traspasos, pattern)
            if col is not None:
                return col
        
        # If not found, print available columns for debugging
        print(f"Available columns in traspasos: {df_traspasos.columns.tolist()}")
        return None

    def get_cantidad_pedida_column(df_productos):
        """
        Get the 'Cantidad Pedida' column from productos DataFrame with robust handling
        """
        if df_productos is None or not isinstance(df_productos, pd.DataFrame):
            return None
        
        # Try multiple patterns to find the column
        patterns = ['Cantidad Pedida', 'CantidadPedida', 'Cantidad_Pedida', 'cantidad pedida', 'cantidadpedida']
        
        for pattern in patterns:
            col = find_column_by_pattern(df_productos, pattern)
            if col is not None:
                return col
        
        # If not found, print available columns for debugging
        print(f"Available columns in productos: {df_productos.columns.tolist()}")
        return None


