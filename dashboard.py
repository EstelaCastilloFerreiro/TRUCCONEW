import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import re  # Add re import for regex

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
        color: #000000;
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
        color: #000000;
        font-size: 14px;
        margin-bottom: 5px;
        margin-top: 0;
    }
    .metric-value {
        color: #000000;
        font-size: 24px;
        font-weight: bold;
        margin: 0;
    }
    .section-title {
        color: #000000;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 20px;
        margin-top: 0;
        line-height: 1.2;
    }
    .viz-title {
        color: #000000;
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
    st.markdown(f"<h4 style='text-align:left;color:#000000;margin:0;padding:0;font-size:20px;font-weight:bold;'>{text}</h4>", unsafe_allow_html=True)

def subtitulo(text):
    st.markdown(f"<h5 style='text-align:left;color:#000000;margin:0;padding:0;font-size:22px;font-weight:bold;'>{text}</h5>", unsafe_allow_html=True)

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
    
    ax.set_title(title, fontsize=16, fontweight='bold', color="#000000", loc="left", pad=0)
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
            color='#000000'
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
    
    df_ventas = df_ventas.copy()
    df_productos = df_productos.copy()
    
    # Asegurarnos que las columnas existen y están en el formato correcto
    df_ventas['Fecha Documento'] = pd.to_datetime(df_ventas['Fecha Documento'], format='%d/%m/%Y', errors='coerce')
    df_ventas = df_ventas.dropna(subset=['Fecha Documento'])

    df_ventas['Mes'] = df_ventas['Fecha Documento'].dt.to_period('M').astype(str)
    df_ventas['Tienda'] = df_ventas['NombreTPV'].astype(str)
    df_ventas['Producto'] = df_ventas['ACT']
    df_ventas['Familia'] = df_ventas['Descripción Familia'].fillna("Sin Familia")
    df_ventas['Cantidad'] = pd.to_numeric(df_ventas['Cantidad'], errors='coerce').fillna(0)
    df_ventas['Descripción Color'] = df_ventas.get('Descripción Color', 'Desconocido')

    # Convertir P.V.P. a numérico en ambos dataframes
    df_ventas['P.V.P.'] = pd.to_numeric(df_ventas['P.V.P.'], errors='coerce').fillna(0)
    df_ventas['Subtotal'] = pd.to_numeric(df_ventas['Subtotal'], errors='coerce').fillna(0)

    # Aplicar filtros
    df_ventas = aplicar_filtros(df_ventas)
    if df_ventas.empty:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        return

    # Calcular ventas en dinero usando Subtotal
    df_ventas['Ventas Dinero'] = df_ventas['Subtotal']

    # Identificar tiendas online y físicas
    df_ventas['Es_Online'] = df_ventas['NombreTPV'].str.contains('ONLINE', case=False, na=False)

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
                            <p style="color: #666666; font-size: 14px; margin: 0 0 5px 0;">Total Ventas</p>
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

            # Par 1: Ventas Mensuales y Top 10 Tiendas con Menos Ventas
            col1, col2 = st.columns(2)
            
            with col1:
                viz_title("Ventas Mensuales por Tipo de Tienda")
                ventas_mes_tipo = df_ventas.groupby(['Mes', 'Es_Online']).agg({
                    'Cantidad': 'sum',
                    'Ventas Dinero': 'sum'
                }).reset_index()
                
                ventas_mes_tipo['Tipo'] = ventas_mes_tipo['Es_Online'].map({True: 'Online', False: 'Física'})
                
                fig = px.bar(ventas_mes_tipo, 
                            x='Mes', 
                            y='Cantidad', 
                            color='Cantidad',
                            color_continuous_scale=COLOR_GRADIENT,
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
                )
                
                fig.update_traces(
                    texttemplate='%{text:,.0f}', 
                    textposition='outside',
                    hovertemplate="Mes: %{x}<br>Cantidad: %{text:,.0f}<br>Ventas: %{customdata:,.2f}€<extra></extra>",
                    customdata=ventas_mes_tipo['Ventas Dinero'],
                    opacity=0.8
                )
                
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                viz_title("Top Tiendas con Menos Ventas")
                ventas_por_tienda = df_ventas.groupby('NombreTPV').agg({
                    'Cantidad': 'sum',
                    'Ventas Dinero': 'sum'
                }).reset_index()
                ventas_por_tienda.columns = ['Tienda', 'Unidades Vendidas', 'Ventas (€)']
                bottom_10_tiendas = ventas_por_tienda.nsmallest(10, 'Unidades Vendidas')
                
                fig = px.bar(
                    bottom_10_tiendas,
                    x='Tienda',
                    y='Unidades Vendidas',
                    text='Ventas (€)',
                    color='Unidades Vendidas',
                    color_continuous_scale=COLOR_GRADIENT,
                    height=400
                )
                fig.update_layout(
                    xaxis_tickangle=45,
                    showlegend=False,
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
                )
                fig.update_traces(
                    texttemplate='%{text:.0f}€', 
                    textposition='outside',
                    opacity=0.8
                )
                st.plotly_chart(fig, use_container_width=True)

            # Par 2: Unidades por Talla y Ventas por Familia
            col3, col4 = st.columns(2)
            
            with col3:
                viz_title("Unidades Vendidas por Talla")
                familias = sorted(df_ventas['Familia'].unique())
                familia_seleccionada = st.selectbox("Selecciona una familia:", familias)
                
                df_familia = df_ventas[df_ventas['Familia'] == familia_seleccionada]
                tallas_familia = df_familia.groupby('Talla')['Cantidad'].sum().reset_index()
                tallas_orden = sorted(tallas_familia['Talla'].unique())
                
                fig = px.bar(
                    tallas_familia,
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
                )
                
                fig.update_traces(
                    texttemplate='%{text:,.0f}', 
                    textposition='outside',
                    opacity=0.8
                )
                st.plotly_chart(fig, use_container_width=True)

            with col4:
                viz_title("Ventas por Familia")
                ventas_familia = df_ventas.groupby('Familia').agg({
                    'Cantidad': 'sum',
                    'Ventas Dinero': 'sum'
                }).reset_index()
                ventas_familia.columns = ['Familia', 'Unidades Vendidas', 'Ventas (€)']
                ventas_familia = ventas_familia.sort_values('Ventas (€)', ascending=True)

                fig = px.bar(
                    ventas_familia,
                    x='Ventas (€)',
                    y='Familia',
                    orientation='h',
                    text='Ventas (€)',
                    color='Ventas (€)',
                    color_continuous_scale=COLOR_GRADIENT,
                    height=400
                )
                fig.update_layout(
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'},
                    margin=dict(t=0, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
                )
                fig.update_traces(
                    texttemplate='%{text:.0f}€', 
                    textposition='outside',
                    opacity=0.8
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error al calcular KPIs: {e}")

    elif seccion == "Geográfico y Tiendas":
        # Preparar datos
        ventas_por_zona = df_ventas.groupby('Zona geográfica')['Cantidad'].sum().reset_index()
        ventas_por_tienda = df_ventas.groupby('Tienda')['Cantidad'].sum().reset_index()
        ventas_por_fam_zona = df_ventas.groupby(['Zona geográfica', 'Familia'])['Cantidad'].sum().reset_index()
        ventas_tienda_mes = df_ventas.groupby(['Mes', 'Tienda'])['Cantidad'].sum().reset_index()
        tiendas_por_zona = df_ventas[['Tienda', 'Zona geográfica']].drop_duplicates().groupby('Zona geográfica').count().reset_index()

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
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="#000000"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Top 5 Tiendas
            top_5_tiendas = ventas_por_tienda.nlargest(5, 'Cantidad')
            fig = px.bar(top_5_tiendas, 
                        x='Tienda', 
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
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="#000000"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        # Segunda fila de visualizaciones
        col3, col4 = st.columns(2)
        
        with col3:
            # Tiendas por Zona
            fig = px.bar(tiendas_por_zona, 
                        x='Zona geográfica', 
                        y='Tienda',
                        color='Tienda',
                        color_continuous_scale=COLOR_GRADIENT,
                        text='Tienda')
            fig.update_layout(
                title="Tiendas por Zona",
                showlegend=False,
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="#000000"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            # Distribución Mensual de Ventas
            pivot_tienda_mes = ventas_tienda_mes.pivot(index='Mes', columns='Tienda', values='Cantidad').fillna(0)
            fig = px.line(pivot_tienda_mes, 
                         color_discrete_sequence=COLOR_GRADIENT)
            fig.update_layout(
                title="Distribución Mensual de Ventas",
                showlegend=True,
                legend_title_text='Tienda',
                xaxis_tickangle=45,
                margin=dict(t=30, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="#000000"
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
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="#000000"
            )
            fig.update_traces(opacity=0.8)
            st.plotly_chart(fig, use_container_width=True)

        with col6:
            # Mapa de Ventas
            df_nacional = df_ventas[~df_ventas['Tienda'].isin(TIENDAS_EXTRANJERAS)].copy()
            df_nacional['Ciudad'] = df_nacional['Tienda'].str.extract(r'ET\d{1,2}-([\w\s\.\(\)]+)')[0]
            df_nacional['Ciudad'] = (
                df_nacional['Ciudad']
                .str.upper()
                .str.replace(r'ECITRUCCO|ECI|XANADU|TRUCCO|CORT.*', '', regex=True)
                .str.strip()
            )

            coordenadas = {
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

            df_nacional['lat'] = df_nacional['Ciudad'].map(lambda c: coordenadas.get(c, (None, None))[0])
            df_nacional['lon'] = df_nacional['Ciudad'].map(lambda c: coordenadas.get(c, (None, None))[1])
            df_nacional = df_nacional.dropna(subset=['lat', 'lon'])

            # Agrupar por ciudad y calcular el total de ventas absolutas
            ventas_ciudad = df_nacional.groupby(['Ciudad', 'lat', 'lon']).agg({
                'Cantidad': lambda x: abs(x).sum(),  # Usar valor absoluto para las cantidades
                'Subtotal': 'sum'
            }).reset_index()

            # Normalizar los tamaños de los marcadores para que sean más manejables
            if not ventas_ciudad.empty:
                # Asegurar que los tamaños sean positivos y estén en un rango razonable
                min_size = 10
                max_size = 50
                ventas_ciudad['marker_size'] = min_size + (ventas_ciudad['Cantidad'] - ventas_ciudad['Cantidad'].min()) / \
                    (ventas_ciudad['Cantidad'].max() - ventas_ciudad['Cantidad'].min()) * (max_size - min_size)

                fig = px.scatter_mapbox(
                    ventas_ciudad,
                    lat='lat',
                    lon='lon',
                    size='marker_size',  # Usar el tamaño normalizado
                    color='Cantidad',
                    hover_name='Ciudad',
                    hover_data={
                        'marker_size': False,  # Ocultar en el hover
                        'Cantidad': True,
                        'Subtotal': ':,.2f€'
                    },
                    color_continuous_scale=COLOR_GRADIENT,
                    zoom=5,
                    height=400
                )
                fig.update_layout(
                    title="Mapa de Ventas",
                    mapbox_style='carto-positron',
                    margin=dict(t=30, b=0, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos disponibles para mostrar en el mapa.")

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
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="#000000"
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
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="#000000"
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
                plot_bgcolor="rgba(0,0,0,0)",
                title_font_color="#000000"
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
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
                    plot_bgcolor="rgba(0,0,0,0)",
                    title_font_color="#000000"
                )
                fig.update_traces(opacity=0.8)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos suficientes para calcular los descuentos.")


