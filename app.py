import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
import sqlite3
import hashlib
import io

# ================================
# CONFIGURACIÓN INICIAL DE LA APP
# ================================
st.set_page_config(
    page_title="Risk Management System",
    layout="wide",
    page_icon="⚠️"
)

# Estilos personalizados (blanco, gris, amarillo)
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
        color: #000000;
    }
    header, .css-18ni7ap.e8zbici2 {
        background-color: #ffffff !important;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .stMetric {
        background: #f7f7f7;
        padding: 1rem;
        border-radius: 0.8rem;
        border: 1px solid #FFD10020;
    }
    </style>
    """,
    unsafe_allow_html=True
)

DB_PATH = "riesgos.db"
TABLE_NAME = "riesgos"

# ======================
# UTILIDADES DE BASE DE DATOS
# ======================
def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """
    Crea las tablas si no existen y asegura columnas nuevas.
    """
    with get_connection() as conn:
        # Tabla de riesgos
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                "Riesgo" TEXT,
                "Probabilidad" REAL,
                "Consecuencia" REAL,
                "Nivel de Riesgo" REAL,
                "Categoría Riesgo" TEXT,
                "Área" TEXT,
                "Responsable" TEXT,
                "Control" TEXT,
                "Fecha" TEXT,
                "Observaciones" TEXT,
                "Origen" TEXT,
                "FileHash" TEXT,
                "Proyecto" TEXT,
                "Estado" TEXT,
                "Criticidad" TEXT,
                "created_at" TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Asegurar columnas nuevas si la tabla ya existía
        columnas_necesarias = ["Riesgo", "Proyecto", "Estado", "Criticidad"]
        cur = conn.execute(f"PRAGMA table_info({TABLE_NAME});")
        existentes = [row[1] for row in cur.fetchall()]

        # Si venías de una versión con "Peligro", renómbrala a "Riesgo" lógicamente
        if "Peligro" in existentes and "Riesgo" not in existentes:
            conn.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN "Riesgo" TEXT;')
            conn.execute(f'UPDATE {TABLE_NAME} SET "Riesgo" = "Peligro";')

        cur = conn.execute(f"PRAGMA table_info({TABLE_NAME});")
        existentes = [row[1] for row in cur.fetchall()]

        for col in columnas_necesarias:
            if col not in existentes:
                conn.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN "{col}" TEXT;')

        # Nueva tabla: planes de tratamiento
        conn.execute("""
            CREATE TABLE IF NOT EXISTS planes_tratamiento (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                riesgo_id INTEGER,
                "Accion" TEXT,
                "Responsable_Accion" TEXT,
                "Fecha_Compromiso" TEXT,
                "Estado_Accion" TEXT,
                "Costo_Estimado" REAL,
                "Comentarios" TEXT,
                "created_at" TEXT DEFAULT CURRENT_TIMESTAMP,
                "updated_at" TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (riesgo_id) REFERENCES riesgos(id)
            );
        """)

        # ==========================
        # TABLAS DE CATÁLOGOS
        # ==========================

        # Catálogo de Riesgos (tipos genéricos de riesgo)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cat_riesgos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT UNIQUE,
                descripcion TEXT,
                activo INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Catálogo de Proyectos / Contratos / Procesos
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cat_proyectos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT UNIQUE,
                codigo TEXT,
                descripcion TEXT,
                activo INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Catálogo de Áreas
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cat_areas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT UNIQUE,
                descripcion TEXT,
                activo INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Catálogo de Responsables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cat_responsables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT UNIQUE,
                cargo TEXT,
                correo TEXT,
                activo INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Catálogo de Controles
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cat_controles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre TEXT UNIQUE,
                descripcion TEXT,
                tipo TEXT,
                activo INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()


@st.cache_data
def cargar_bd():
    """
    Carga TODA la base de datos de riesgos desde SQLite.
    Si no hay datos, retorna un DF vacío con la estructura estándar.
    """
    columnas = [
        "id", "Riesgo", "Probabilidad", "Consecuencia", "Nivel de Riesgo",
        "Categoría Riesgo", "Área", "Responsable", "Control",
        "Fecha", "Observaciones", "Origen", "FileHash",
        "Proyecto", "Estado", "Criticidad", "created_at"
    ]
    try:
        with get_connection() as conn:
            df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
        # Asegurar todas las columnas
        for col in columnas:
            if col not in df.columns:
                df[col] = None
        df = df[columnas]
        return df
    except Exception:
        return pd.DataFrame(columns=columnas)


def limpiar_cache_bd():
    cargar_bd.clear()


def insertar_df_en_bd(df_bd: pd.DataFrame):
    """Inserta un DataFrame con las columnas de la matriz en la tabla riesgos."""
    if df_bd.empty:
        return
    with get_connection() as conn:
        df_bd.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        conn.commit()
    limpiar_cache_bd()


def cargar_planes():
    try:
        with get_connection() as conn:
            df = pd.read_sql("SELECT * FROM planes_tratamiento", conn)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "id", "riesgo_id", "Accion", "Responsable_Accion",
            "Fecha_Compromiso", "Estado_Accion", "Costo_Estimado",
            "Comentarios", "created_at", "updated_at"
        ])


def insertar_plan(df_plan: pd.DataFrame):
    if df_plan.empty:
        return
    with get_connection() as conn:
        df_plan.to_sql("planes_tratamiento", conn, if_exists="append", index=False)
        conn.commit()


# ======================
# UTILIDADES PARA CATÁLOGOS
# ======================
@st.cache_data
def cargar_catalogo(nombre_tabla: str) -> pd.DataFrame:
    try:
        with get_connection() as conn:
            df = pd.read_sql(f"SELECT * FROM {nombre_tabla}", conn)
        return df
    except Exception:
        return pd.DataFrame()


def limpiar_cache_catalogos():
    cargar_catalogo.clear()


def insertar_en_catalogo(nombre_tabla: str, data: dict):
    df = pd.DataFrame([data])
    with get_connection() as conn:
        df.to_sql(nombre_tabla, conn, if_exists="append", index=False)
        conn.commit()
    limpiar_cache_catalogos()


# ======================
# FUNCIONES AUXILIARES DE RIESGO
# ======================
def get_column(df, target_name):
    """Devuelve el nombre real de la columna en el DataFrame, buscando por coincidencia en minúsculas."""
    cols_lower = {c.lower(): c for c in df.columns}
    return cols_lower.get(target_name.lower())


def calcular_nivel_riesgo(df, col_prob, col_cons, col_nivel=None):
    """Calcula o recalcula el Nivel de Riesgo = Probabilidad x Consecuencia."""
    if col_nivel is None or col_nivel not in df.columns:
        df["Nivel de Riesgo"] = df[col_prob] * df[col_cons]
        col_nivel = "Nivel de Riesgo"
    else:
        df[col_nivel] = df[col_prob] * df[col_cons]
    return df, col_nivel


def categorizar_riesgo(valor):
    """
    Lógica estándar:
      - Bajo: 1–7
      - Medio: 8–14
      - Alto: 15–25
    """
    try:
        v = float(valor)
    except (ValueError, TypeError):
        return "Sin dato"
    if v >= 15:
        return "Alto"
    elif v >= 8:
        return "Medio"
    elif v > 0:
        return "Bajo"
    else:
        return "Sin dato"


def detectar_columna_fecha(df):
    """Intenta detectar automáticamente una columna de fecha."""
    candidatos_preferidos = ["fecha", "fecha evento", "date", "fecha_riesgo"]
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in candidatos_preferidos:
        if cand in cols_lower:
            col = cols_lower[cand]
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().any():
                    return col, df
            except Exception:
                pass

    datetime_cols = [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if datetime_cols:
        return datetime_cols[0], df

    for c in df.columns:
        if df[c].dtype == object:
            try:
                converted = pd.to_datetime(df[c], errors="coerce")
                if converted.notna().sum() > 0:
                    df[c] = converted
                    return c, df
            except Exception:
                continue

    return None, df


def detectar_fila_encabezado(df_raw, nombres_minimos, max_filas_busqueda=15):
    """
    Detecta la fila que parece ser el encabezado de la matriz,
    buscando al menos 3 coincidencias de nombres esperados.
    """
    max_filas = min(max_filas_busqueda, len(df_raw))
    targets = [n.lower() for n in nombres_minimos]

    for i in range(max_filas):
        valores = df_raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        score = sum(1 for t in targets if t in valores)
        if score >= 3:
            return i
    return None


def calcular_kpis_y_graficos(df, col_area="Área", titulo_prefix=""):
    """KPIs + gráficos + matriz P×C + tendencias + tabla a partir de un DataFrame de riesgos."""
    if df.empty:
        st.info("No hay datos para mostrar aún.")
        return

    # KPIs
    total_riesgos = len(df)
    total_alto = (df["Categoría Riesgo"] == "Alto").sum()
    total_medio = (df["Categoría Riesgo"] == "Medio").sum()
    total_bajo = (df["Categoría Riesgo"] == "Bajo").sum()

    col1, col2, col3, col4 = st.columns(4)

    # Bloque total (neutral)
    with col1:
        st.markdown(
            f"""
            <div style="
                background-color:#f7f7f7;
                padding:1rem;
                border-radius:0.8rem;
                border:1px solid #FFD10020;
                text-align:center;
            ">
                <div style="font-size:0.9rem;font-weight:600;">Total de Riesgos</div>
                <div style="font-size:1.8rem;font-weight:800;">{total_riesgos}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Riesgos Altos – rojo
    with col2:
        bg_alto = "#b71c1c" if total_alto > 0 else "#3a3a3a"
        st.markdown(
            f"""
            <div style="
                background-color:{bg_alto};
                padding:1rem;
                border-radius:0.8rem;
                border:1px solid #FFD10040;
                text-align:center;
            ">
                <div style="font-size:0.9rem;font-weight:600;">Riesgos Altos</div>
                <div style="font-size:1.8rem;font-weight:800;">{total_alto}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Riesgos Medios – amarillo
    with col3:
        bg_medio = "#FFD100" if total_medio > 0 else "#d1b800"
        texto_medio_color = "#000000"
        st.markdown(
            f"""
            <div style="
                background-color:{bg_medio};
                padding:1rem;
                border-radius:0.8rem;
                border:1px solid #FFD10060;
                text-align:center;
                color:{texto_medio_color};
            ">
                <div style="font-size:0.9rem;font-weight:600;">Riesgos Medios</div>
                <div style="font-size:1.8rem;font-weight:800;">{total_medio}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Riesgos Bajos – verde
    with col4:
        bg_bajo = "#1b5e20" if total_bajo > 0 else "#2f4f2f"
        st.markdown(
            f"""
            <div style="
                background-color:{bg_bajo};
                padding:1rem;
                border-radius:0.8rem;
                border:1px solid #00e67640;
                text-align:center;
                color:#ffffff;
            ">
                <div style="font-size:0.9rem;font-weight:600;">Riesgos Bajos</div>
                <div style="font-size:1.8rem;font-weight:800;">{total_bajo}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    # Gráfico de barras por área
    with col_g1:
        st.subheader(f"📊 {titulo_prefix}Riesgos por Área")
        if col_area in df.columns and not df[col_area].dropna().empty:
            riesgos_area = (
                df
                .groupby(col_area)
                .size()
                .reset_index(name="Total Riesgos")
            )
            fig_bar = px.bar(
                riesgos_area,
                x=col_area,
                y="Total Riesgos",
                text="Total Riesgos",
                title=f"{titulo_prefix}Distribución de Riesgos por Área",
                color_discrete_sequence=["#FFD100"]
            )
            fig_bar.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#000000",
                xaxis_title="Área",
                yaxis_title="Número de riesgos"
            )
            st.plotly_chart(fig_bar, width='stretch')
        else:
            st.write("Sin datos de Área para mostrar.")

    # Gráfico de pastel por categoría
    with col_g2:
        st.subheader(f"🥧 {titulo_prefix}Distribución de Categorías de Riesgo")
        if "Categoría Riesgo" in df.columns and not df["Categoría Riesgo"].dropna().empty:
            dist_cat = (
                df
                .groupby("Categoría Riesgo")
                .size()
                .reset_index(name="Total")
            )
            color_map = {
                "Alto": "#b71c1c",
                "Medio": "#FFD100",
                "Bajo": "#1b5e20",
                "Sin dato": "#4d4d4d"
            }
            fig_pie = px.pie(
                dist_cat,
                values="Total",
                names="Categoría Riesgo",
                title=f"{titulo_prefix}Distribución de Riesgos por Categoría",
                color="Categoría Riesgo",
                color_discrete_map=color_map
            )
            fig_pie.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#000000",
            )
            st.plotly_chart(fig_pie, width='stretch')
        else:
            st.write("Sin datos de categoría para mostrar.")

    # 🔥 Matriz de calor Probabilidad x Consecuencia (1–5) con riesgos por celda
    st.subheader(f"🧱 {titulo_prefix}Matriz de calor Probabilidad x Consecuencia")

    if "Probabilidad" in df.columns and "Consecuencia" in df.columns:
        # Copia y normalización
        df_pc = df.copy()
        df_pc["Probabilidad"] = pd.to_numeric(df_pc["Probabilidad"], errors="coerce")
        df_pc["Consecuencia"] = pd.to_numeric(df_pc["Consecuencia"], errors="coerce")
        df_pc = df_pc.dropna(subset=["Probabilidad", "Consecuencia"])

        # Rango estándar 1–5 en ambos ejes
        probs = range(1, 6)
        consec = range(1, 6)

        # Matrices:
        # - matriz_cat_val: 1 = Bajo, 2 = Medio, 3 = Alto (para color)
        # - matriz_text: texto a mostrar en cada celda (P×C y nº de riesgos)
        # - matriz_hover: texto detallado con nombres de riesgos
        matriz_cat_val = []
        matriz_text = []
        matriz_hover = []

        cat_to_val = {"Bajo": 1, "Medio": 2, "Alto": 3}

        for p in probs:
            fila_cat = []
            fila_text = []
            fila_hover = []
            for c in consec:
                prod = int(p * c)
                nivel = categorizar_riesgo(prod)  # Bajo / Medio / Alto

                # Riesgos que caen exactamente en esta combinación P,C
                riesgos_match = df_pc[
                    (df_pc["Probabilidad"] == p) &
                    (df_pc["Consecuencia"] == c)
                ]

                lista_riesgos = (
                    riesgos_match.get("Riesgo", pd.Series([], dtype=str))
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                # Texto visible en la celda: "P×C (nR)" si hay riesgos, solo "P×C" si no
                if lista_riesgos:
                    cell_text = f"{prod} ({len(lista_riesgos)}R)"
                    resumen = "; ".join(lista_riesgos[:3])
                    if len(lista_riesgos) > 3:
                        resumen += "..."
                else:
                    cell_text = f"{prod}"
                    resumen = "Sin riesgos en esta combinación"

                hover_txt = (
                    f"Probabilidad: {p}<br>"
                    f"Consecuencia: {c}<br>"
                    f"P × C: {prod}<br>"
                    f"Nivel: {nivel}<br>"
                    f"Riesgos: {resumen}"
                )

                fila_cat.append(cat_to_val.get(nivel, 0))
                fila_text.append(cell_text)
                fila_hover.append(hover_txt)

            matriz_cat_val.append(fila_cat)
            matriz_text.append(fila_text)
            matriz_hover.append(fila_hover)

        fig_heat = px.imshow(
            matriz_cat_val,
            x=list(consec),
            y=list(probs),
            aspect="auto",
            labels={"x": "Consecuencia", "y": "Probabilidad", "color": "Nivel"},
            title=f"{titulo_prefix}Matriz de calor P × C (1–5)"
        )

        # Colores: Bajo = verde, Medio = amarillo, Alto = rojo
        colorscale = [
            (0.0, "#1b5e20"),   # Bajo - verde
            (1/3, "#1b5e20"),
            (1/3 + 1e-6, "#FFD100"),  # Medio - amarillo
            (2/3, "#FFD100"),
            (2/3 + 1e-6, "#b71c1c"),  # Alto - rojo
            (1.0, "#b71c1c"),
        ]

        fig_heat.update_traces(
            text=matriz_text,
            texttemplate="%{text}",
            hovertext=matriz_hover,
            hovertemplate="%{hovertext}<extra></extra>"
        )

        fig_heat.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font_color="#000000",
            xaxis_title="Consecuencia",
            yaxis_title="Probabilidad",
            coloraxis=dict(
                colorscale=colorscale,
                cmin=1,
                cmax=3,
                colorbar=dict(
                    tickmode="array",
                    tickvals=[1, 2, 3],
                    ticktext=["Bajo", "Medio", "Alto"],
                    title="Nivel de riesgo"
                )
            )
        )

        st.plotly_chart(fig_heat, width='stretch')

    else:
        st.info("La base no tiene columnas 'Probabilidad' y 'Consecuencia' para construir la matriz de calor.")

    # Tendencias en el tiempo usando columna Fecha
    if "Fecha" in df.columns and df["Fecha"].notna().any():
        df_fecha = df.copy()
        try:
            df_fecha["Fecha"] = pd.to_datetime(df_fecha["Fecha"], errors="coerce")
            df_fecha = df_fecha.dropna(subset=["Fecha"])
            if not df_fecha.empty:
                df_fecha["Fecha"] = df_fecha["Fecha"].dt.date
                tendencia = (
                    df_fecha
                    .groupby("Fecha")
                    .size()
                    .reset_index(name="Total Riesgos")
                )
                st.subheader(f"📈 {titulo_prefix}Tendencia de riesgos en el tiempo")
                fig_line = px.line(
                    tendencia,
                    x="Fecha",
                    y="Total Riesgos",
                    markers=True,
                    title=f"{titulo_prefix}Tendencia de aparición de riesgos"
                )
                fig_line.update_layout(
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font_color="#000000",
                    xaxis_title="Fecha",
                    yaxis_title="Número de riesgos"
                )
                st.plotly_chart(fig_line, width='stretch')
        except Exception:
            st.info("No fue posible procesar la columna Fecha para tendencias.")
    else:
        st.info("No se encontró columna de fecha para mostrar tendencias.")

    st.markdown("---")

    # Tabla
    st.subheader(f"📋 {titulo_prefix}Tabla de riesgos")
    st.dataframe(df, width='stretch', height=400)


def integrar_archivo_a_bd(file_bytes: bytes, filename: str):
    """
    Lee un archivo (Excel/CSV), detecta encabezados, normaliza columnas
    y lo integra como registros en la base de datos global.
    Acepta encabezado 'Riesgo' o 'Peligro', pero lo mapea a 'Riesgo'.
    """
    file_hash = hashlib.md5(file_bytes).hexdigest()

    # Verificar si ese hash ya existe en la BD
    df_bd = cargar_bd()
    if "FileHash" in df_bd.columns and not df_bd.empty:
        if file_hash in df_bd["FileHash"].dropna().unique().tolist():
            st.info("Este archivo ya fue integrado previamente a la base de datos global.")
            return

    # Leer sin encabezados
    if filename.lower().endswith(".csv"):
        df_raw = pd.read_csv(io.BytesIO(file_bytes), header=None)
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)

    nombres_minimos = ["Riesgo", "Peligro", "Probabilidad", "Consecuencia", "Área", "Area", "Responsable", "Control"]
    fila_header = detectar_fila_encabezado(df_raw, nombres_minimos)

    if fila_header is None:
        st.error(
            "❌ No se pudo detectar la fila de encabezados. "
            "Verifica que tu archivo tenga columnas como `Riesgo` (o `Peligro`), "
            "`Probabilidad`, `Consecuencia`, `Área`, `Responsable`, `Control` en alguna de las primeras filas."
        )
        st.write("Primeras filas detectadas:")
        st.dataframe(df_raw.head(10), width='stretch')
        return

    header_row = df_raw.iloc[fila_header].astype(str).str.strip()
    df = df_raw.iloc[fila_header + 1:].copy()
    df.columns = header_row
    df = df.reset_index(drop=True)
    df.columns = df.columns.map(lambda x: str(x).strip())

    # Identificar columnas clave
    col_riesgo = get_column(df, "Riesgo") or get_column(df, "Peligro")
    col_prob = get_column(df, "Probabilidad")
    col_cons = get_column(df, "Consecuencia")
    col_nivel = get_column(df, "Nivel de Riesgo")
    col_area = get_column(df, "Área") or get_column(df, "Area")
    col_resp = get_column(df, "Responsable")
    col_control = get_column(df, "Control")
    col_obs = get_column(df, "Observaciones")

    col_proy = get_column(df, "Proyecto") or get_column(df, "Contrato") or get_column(df, "Proceso")
    col_estado = get_column(df, "Estado")
    col_criticidad = get_column(df, "Criticidad")

    columnas_requeridas = [col_riesgo, col_prob, col_cons, col_area, col_resp, col_control]

    if any(c is None for c in columnas_requeridas):
        st.error(
            "❌ Faltan columnas requeridas incluso después de detectar encabezados.\n\n"
            "Verifica que existan estas columnas (respetando tildes y ortografía): "
            "`Riesgo` o `Peligro`, `Probabilidad`, `Consecuencia`, `Área` o `Area`, `Responsable`, `Control`."
        )
        st.write("Columnas detectadas:", list(df.columns))
        return

    # Asegurar numérico
    df[col_prob] = pd.to_numeric(df[col_prob], errors="coerce")
    df[col_cons] = pd.to_numeric(df[col_cons], errors="coerce")

    # Calcular nivel de riesgo y categoría
    df, col_nivel = calcular_nivel_riesgo(df, col_prob, col_cons, col_nivel)
    df["Categoría Riesgo"] = df[col_nivel].apply(categorizar_riesgo)

    # Fecha (si tiene)
    col_fecha, df = detectar_columna_fecha(df)

    fecha_vals = df[col_fecha] if col_fecha is not None else None
    if fecha_vals is not None:
        fecha_vals = fecha_vals.dt.strftime("%Y-%m-%d")

    if col_obs is not None:
        obs_vals = df[col_obs]
    else:
        obs_vals = ""

    proy_vals = df[col_proy] if col_proy is not None else ""
    estado_vals = df[col_estado] if col_estado is not None else "Identificado"
    criticidad_vals = df[col_criticidad] if col_criticidad is not None else ""

    df_para_bd = pd.DataFrame({
        "Riesgo": df[col_riesgo],
        "Probabilidad": df[col_prob],
        "Consecuencia": df[col_cons],
        "Nivel de Riesgo": df[col_nivel],
        "Categoría Riesgo": df["Categoría Riesgo"],
        "Área": df[col_area],
        "Responsable": df[col_resp],
        "Control": df[col_control],
        "Fecha": fecha_vals,
        "Observaciones": obs_vals,
        "Origen": "Archivo",
        "FileHash": file_hash,
        "Proyecto": proy_vals,
        "Estado": estado_vals,
        "Criticidad": criticidad_vals
    })

    insertar_df_en_bd(df_para_bd)
    st.success("✅ Archivo integrado exitosamente a la base de datos global de riesgos.")


# ======================
# INICIO APP
# ======================
init_db()

with st.sidebar:
    st.markdown("## ⚙️ Navegación")
    vista = st.radio(
        "Selecciona la vista:",
        [
            "Dashboard de la base de datos",
            "Formulario en tiempo real",
            "Gestión y actualización",
            "Planes de tratamiento",
            "Administración de catálogos"
        ],
        index=0
    )

# ----------------------------------------
# VISTA 1: DASHBOARD SOBRE BASE DE DATOS GLOBAL
# ----------------------------------------
if vista == "Dashboard de la base de datos":
    st.title("⚠️ Dashboard de Riesgos")
    st.caption(
        "Analítica sobre la base de datos global de riesgos (SQLite). "
        "Los archivos y el formulario solo alimentan esta base; todo lo que ves aquí viene de la BD global."
    )

    # Carga de archivo e integración
    with st.sidebar:
        st.markdown("---")
        st.header("📂 Carga e integración de archivo")
        uploaded_file = st.file_uploader(
            "Sube tu matriz de riesgos (Excel/CSV)",
            type=["xlsx", "xls", "csv"],
            key="upload_dashboard"
        )
        st.write(
            "Columnas mínimas esperadas:\n"
            "- Riesgo (o Peligro)\n- Probabilidad\n- Consecuencia\n- Área\n"
            "- Responsable\n- Control\n- (Opcional) Nivel de Riesgo\n- (Opcional) Fecha\n"
            "Al cargar, los registros se integran a la base de datos global."
        )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        integrar_archivo_a_bd(file_bytes, uploaded_file.name)

    # Filtros y analítica sobre BD global
    bd_global = cargar_bd()

    with st.sidebar:
        st.markdown("---")
        st.header("🎛️ Filtros sobre la base de datos global")
        if bd_global.empty:
            st.info("Aún no hay datos en la base de datos global.")
            filter_area = filter_resp = filter_cat = filter_origen = []
            search_text = ""
        else:
            areas = sorted(bd_global["Área"].dropna().unique().tolist())
            responsables = sorted(bd_global["Responsable"].dropna().unique().tolist())
            categorias = sorted(bd_global["Categoría Riesgo"].dropna().unique().tolist())
            origenes = sorted(bd_global["Origen"].dropna().unique().tolist())

            filter_area = st.multiselect("Filtrar por Área", options=areas, default=areas)
            filter_resp = st.multiselect("Filtrar por Responsable", options=responsables, default=responsables)
            filter_cat = st.multiselect("Filtrar por Categoría de Riesgo", options=categorias, default=categorias)
            filter_origen = st.multiselect("Filtrar por Origen", options=origenes, default=origenes)
            search_text = st.text_input("🔍 Buscar (Riesgo / Control)", "")

    st.markdown("### 📚 Base de datos global de riesgos")

    if bd_global.empty:
        st.info("No hay registros aún. Carga un archivo o registra riesgos en el formulario.")
    else:
        df_filtrado = bd_global.copy()

        # Filtros
        if filter_area:
            df_filtrado = df_filtrado[df_filtrado["Área"].isin(filter_area)]
        if filter_resp:
            df_filtrado = df_filtrado[df_filtrado["Responsable"].isin(filter_resp)]
        if filter_cat:
            df_filtrado = df_filtrado[df_filtrado["Categoría Riesgo"].isin(filter_cat)]
        if filter_origen:
            df_filtrado = df_filtrado[df_filtrado["Origen"].isin(filter_origen)]

        if search_text:
            mask = (
                df_filtrado["Riesgo"].astype(str).str.contains(search_text, case=False, na=False) |
                df_filtrado["Control"].astype(str).str.contains(search_text, case=False, na=False)
            )
            df_filtrado = df_filtrado[mask]

        # Dashboard sobre BD global filtrada
        calcular_kpis_y_graficos(df_filtrado, col_area="Área", titulo_prefix="BD Global - ")

        # Export general
        st.markdown("### 📤 Exportar datos filtrados para Power BI / Excel")
        csv_data = df_filtrado.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="💾 Descargar CSV filtrado",
            data=csv_data,
            file_name="matriz_riesgos_bd_filtrada.csv",
            mime="text/csv"
        )

        # Export directo de Riesgos Altos
        st.markdown("### 🚨 Exportar solo Riesgos Altos")
        df_altos = df_filtrado[df_filtrado["Categoría Riesgo"] == "Alto"].copy()

        if df_altos.empty:
            st.info("No hay riesgos altos en el filtro actual.")
        else:
            csv_altos = df_altos.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="💾 Descargar solo Riesgos Altos (CSV)",
                data=csv_altos,
                file_name="matriz_riesgos_altos_filtrada.csv",
                mime="text/csv"
            )

# ----------------------------------------
# VISTA 2: FORMULARIO EN TIEMPO REAL
# ----------------------------------------
elif vista == "Formulario en tiempo real":
    st.title("📝 Registro en tiempo real de riesgos")
    st.caption("Los riesgos registrados aquí también se integran a la misma base de datos global (persistente en SQLite).")

    with st.expander("📁 Descargar plantilla de matriz de riesgos (CSV)", expanded=False):
        plantilla = pd.DataFrame(columns=[
            "Riesgo",
            "Probabilidad",
            "Consecuencia",
            "Nivel de Riesgo",
            "Categoría Riesgo",
            "Área",
            "Responsable",
            "Control",
            "Fecha",
            "Observaciones",
            "Origen",
            "FileHash",
            "Proyecto",
            "Estado",
            "Criticidad"
        ])
        csv_plantilla = plantilla.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "💾 Descargar plantilla CSV",
            data=csv_plantilla,
            file_name="plantilla_matriz_riesgos.csv",
            mime="text/csv"
        )

    st.markdown("### ➕ Agregar nuevo riesgo")

    with st.form("form_riesgo"):
        col_a, col_b = st.columns(2)
        with col_a:
            riesgo = st.text_input("Riesgo", "")
            proyecto = st.text_input("Proyecto / Contrato / Proceso", "")
            area = st.text_input("Área", "")
            responsable = st.text_input("Responsable", "")
            control = st.text_area("Control existente / medida de mitigación", "")
        with col_b:
            probabilidad = st.number_input("Probabilidad (1–5)", min_value=1, max_value=5, value=3, step=1)
            consecuencia = st.number_input("Consecuencia (1–5)", min_value=1, max_value=5, value=3, step=1)
            fecha_riesgo = st.date_input("Fecha del riesgo", value=date.today())
            estado = st.selectbox("Estado del riesgo", ["Identificado", "Evaluado", "Tratado", "Cerrado"])
            criticidad = st.selectbox("Criticidad", ["", "Crítico", "Importante", "Moderado", "Menor"])
            observaciones = st.text_area("Observaciones", "")

        submitted = st.form_submit_button("✅ Registrar riesgo")

    if submitted:
        nivel = probabilidad * consecuencia
        categoria = categorizar_riesgo(nivel)

        df_nuevo = pd.DataFrame([{
            "Riesgo": riesgo,
            "Probabilidad": probabilidad,
            "Consecuencia": consecuencia,
            "Nivel de Riesgo": nivel,
            "Categoría Riesgo": categoria,
            "Área": area,
            "Responsable": responsable,
            "Control": control,
            "Fecha": fecha_riesgo.strftime("%Y-%m-%d"),
            "Observaciones": observaciones,
            "Origen": "Formulario",
            "FileHash": None,
            "Proyecto": proyecto,
            "Estado": estado,
            "Criticidad": criticidad
        }])

        insertar_df_en_bd(df_nuevo)
        st.success("✅ Riesgo registrado en la base de datos global.")

    st.markdown("---")
    st.subheader("📚 Base de datos global de riesgos")

    bd = cargar_bd()

    if bd.empty:
        st.info("Aún no has registrado riesgos. Usa el formulario de arriba o la carga por archivo.")
    else:
        calcular_kpis_y_graficos(bd, col_area="Área", titulo_prefix="BD Global - ")

        st.markdown("### 📤 Exportar base de datos completa a Power BI / Excel")
        csv_bd = bd.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="💾 Descargar base de datos completa CSV",
            data=csv_bd,
            file_name="matriz_riesgos_bd_global.csv",
            mime="text/csv"
        )

# ----------------------------------------
# VISTA 3: GESTIÓN Y ACTUALIZACIÓN
# ----------------------------------------
elif vista == "Gestión y actualización":
    st.title("🛠 Gestión y actualización de riesgos")
    st.caption("Edita el estado, área, responsable, controles y atributos clave de los riesgos existentes.")

    bd = cargar_bd()

    if bd.empty:
        st.info("No hay datos en la base de datos global todavía.")
    else:
        # Filtro rápido por proyecto y estado
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            proyectos = sorted([p for p in bd["Proyecto"].dropna().unique().tolist() if p != ""])
            proyecto_sel = st.selectbox("Filtrar por proyecto", ["(Todos)"] + proyectos)
        with col_f2:
            estados = sorted([e for e in bd["Estado"].dropna().unique().tolist() if e != ""])
            estado_sel = st.selectbox("Filtrar por estado", ["(Todos)"] + estados)
        with col_f3:
            crits = sorted([c for c in bd["Criticidad"].dropna().unique().tolist() if c != ""])
            crit_sel = st.selectbox("Filtrar por criticidad", ["(Todos)"] + crits)

        df_gestion = bd.copy()
        if proyecto_sel != "(Todos)":
            df_gestion = df_gestion[df_gestion["Proyecto"] == proyecto_sel]
        if estado_sel != "(Todos)":
            df_gestion = df_gestion[df_gestion["Estado"] == estado_sel]
        if crit_sel != "(Todos)":
            df_gestion = df_gestion[df_gestion["Criticidad"] == crit_sel]

        st.markdown("### Selecciona un riesgo para editar")

        if df_gestion.empty:
            st.info("No hay riesgos que cumplan los filtros seleccionados.")
        else:
            df_select = df_gestion[["id", "Proyecto", "Área", "Riesgo", "Estado", "Criticidad"]].copy()
            df_select["label"] = df_select.apply(
                lambda r: f'[{int(r["id"])}] {str(r["Proyecto"] or "").upper()} - {str(r["Riesgo"])[0:50]}...',
                axis=1
            )

            opciones = df_select["label"].tolist()
            map_id = dict(zip(df_select["label"], df_select["id"]))

            seleccion = st.selectbox("Riesgo", opciones)
            riesgo_id = map_id[seleccion]

            riesgo_row = bd[bd["id"] == riesgo_id].iloc[0]

            st.markdown("### Detalle del riesgo seleccionado")
            st.write(f"**ID:** {riesgo_id}")
            st.write(f"**Riesgo:** {riesgo_row['Riesgo']}")
            st.write(f"**Área:** {riesgo_row['Área']}")
            st.write(f"**Proyecto:** {riesgo_row['Proyecto']}")
            st.write(f"**Origen:** {riesgo_row['Origen']}")

            st.markdown("### Actualizar campos clave")

            estados_posibles = ["Identificado", "Evaluado", "Tratado", "Cerrado"]
            crit_posibles = ["", "Crítico", "Importante", "Moderado", "Menor"]

            try:
                idx_estado = estados_posibles.index(riesgo_row["Estado"]) if riesgo_row["Estado"] in estados_posibles else 0
            except Exception:
                idx_estado = 0

            try:
                idx_crit = crit_posibles.index(riesgo_row["Criticidad"]) if riesgo_row["Criticidad"] in crit_posibles else 0
            except Exception:
                idx_crit = 0

            with st.form("form_update_riesgo"):
                col_u1, col_u2 = st.columns(2)
                with col_u1:
                    nueva_area = st.text_input("Área", value=riesgo_row["Área"] or "")
                    nuevo_responsable = st.text_input("Responsable", value=riesgo_row["Responsable"] or "")
                    nuevo_estado = st.selectbox(
                        "Estado",
                        estados_posibles,
                        index=idx_estado
                    )
                    nueva_criticidad = st.selectbox(
                        "Criticidad",
                        crit_posibles,
                        index=idx_crit
                    )
                with col_u2:
                    nuevo_control = st.text_area(
                        "Control / acción de tratamiento",
                        value=riesgo_row["Control"] or "",
                        height=120
                    )
                    nuevas_obs = st.text_area(
                        "Observaciones",
                        value=riesgo_row["Observaciones"] or "",
                        height=120
                    )

                actualizar = st.form_submit_button("💾 Guardar cambios")

            if actualizar:
                with get_connection() as conn:
                    conn.execute(
                        f"""
                        UPDATE {TABLE_NAME}
                        SET "Área" = ?,
                            "Responsable" = ?,
                            "Estado" = ?,
                            "Criticidad" = ?,
                            "Control" = ?,
                            "Observaciones" = ?
                        WHERE id = ?;
                        """,
                        (
                            nueva_area,
                            nuevo_responsable,
                            nuevo_estado,
                            nueva_criticidad,
                            nuevo_control,
                            nuevas_obs,
                            int(riesgo_id)
                        )
                    )
                    conn.commit()
                limpiar_cache_bd()
                st.success("✅ Riesgo actualizado correctamente.")

# ----------------------------------------
# VISTA 4: PLANES DE TRATAMIENTO
# ----------------------------------------
elif vista == "Planes de tratamiento":
    st.title("🛡 Planes de tratamiento de riesgos")
    st.caption("Define, asigna y monitorea acciones de tratamiento para cada riesgo.")

    bd = cargar_bd()

    if bd.empty:
        st.info("No hay riesgos registrados aún. Registra riesgos antes de crear planes.")
    else:
        # Seleccionar riesgo
        st.markdown("### Selecciona un riesgo para asociar un plan de tratamiento")

        df_select = bd[["id", "Proyecto", "Área", "Riesgo", "Estado", "Criticidad"]].copy()
        df_select["label"] = df_select.apply(
            lambda r: f'[{int(r["id"])}] {str(r["Proyecto"] or "").upper()} - {str(r["Riesgo"])[0:50]}...',
            axis=1
        )

        opciones = df_select["label"].tolist()
        map_id = dict(zip(df_select["label"], df_select["id"]))

        if not opciones:
            st.info("No hay riesgos para listar.")
        else:
            seleccion = st.selectbox("Riesgo", opciones)
            riesgo_id = map_id[seleccion]

            riesgo_row = bd[bd["id"] == riesgo_id].iloc[0]

            st.write(f"**Riesgo:** {riesgo_row['Riesgo']}")
            st.write(f"**Proyecto:** {riesgo_row['Proyecto']}")
            st.write(f"**Área:** {riesgo_row['Área']}")
            st.write(f"**Estado actual del riesgo:** {riesgo_row['Estado']}")
            st.write(f"**Criticidad:** {riesgo_row['Criticidad']}")

            st.markdown("### Crear nuevo plan de tratamiento")

            with st.form("form_plan"):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    accion = st.text_area("Acción de tratamiento", "")
                    responsable_accion = st.text_input("Responsable de la acción", riesgo_row["Responsable"] or "")
                    estado_accion = st.selectbox(
                        "Estado de la acción",
                        ["Pendiente", "En ejecución", "Completada", "Retrasada"]
                    )
                with col_p2:
                    fecha_compromiso = st.date_input("Fecha compromiso", value=date.today())
                    costo_estimado = st.number_input("Costo estimado", min_value=0.0, step=100000.0, format="%.0f")
                    comentarios = st.text_area("Comentarios / notas de seguimiento", "")

                submitted_plan = st.form_submit_button("✅ Registrar plan de tratamiento")

            if submitted_plan:
                df_plan = pd.DataFrame([{
                    "riesgo_id": int(riesgo_id),
                    "Accion": accion,
                    "Responsable_Accion": responsable_accion,
                    "Fecha_Compromiso": fecha_compromiso.strftime("%Y-%m-%d"),
                    "Estado_Accion": estado_accion,
                    "Costo_Estimado": costo_estimado,
                    "Comentarios": comentarios
                }])
                insertar_plan(df_plan)
                st.success("✅ Plan de tratamiento registrado para este riesgo.")

            # Listado de planes existentes para ese riesgo
            st.markdown("---")
            st.markdown("### Planes registrados para este riesgo")

            df_planes = cargar_planes()
            df_planes_riesgo = df_planes[df_planes["riesgo_id"] == int(riesgo_id)]

            if df_planes_riesgo.empty:
                st.info("Este riesgo aún no tiene planes de tratamiento registrados.")
            else:
                st.dataframe(df_planes_riesgo, width='stretch', height=300)

# ----------------------------------------
# VISTA 5: ADMINISTRACIÓN DE CATÁLOGOS
# ----------------------------------------
elif vista == "Administración de catálogos":
    st.title("🧩 Administración de catálogos")
    st.caption("Configura los maestros de datos que alimentan tu Risk Management System.")

    tabs = st.tabs([
        "Catálogo de Riesgos",
        "Proyectos / Contratos / Procesos",
        "Áreas",
        "Responsables",
        "Controles"
    ])

    # 1) Catálogo de Riesgos
    with tabs[0]:
        st.subheader("📌 Catálogo de Riesgos")

        with st.form("form_cat_riesgos"):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("Nombre del riesgo (tipo)", "")
            with col2:
                descripcion = st.text_area("Descripción del riesgo", "", height=80)

            submitted = st.form_submit_button("✅ Guardar riesgo en catálogo")

        if submitted:
            if nombre.strip() == "":
                st.warning("El nombre del riesgo no puede estar vacío.")
            else:
                try:
                    insertar_en_catalogo("cat_riesgos", {
                        "nombre": nombre.strip(),
                        "descripcion": descripcion.strip()
                    })
                    st.success("✅ Riesgo agregado al catálogo.")
                except Exception as e:
                    st.error(f"No se pudo guardar el riesgo. Detalle: {e}")

        st.markdown("### 📋 Riesgos en catálogo")
        df_cat = cargar_catalogo("cat_riesgos")
        if df_cat.empty:
            st.info("Aún no hay riesgos en el catálogo.")
        else:
            st.dataframe(df_cat, width='stretch', height=300)

    # 2) Proyectos / Contratos / Procesos
    with tabs[1]:
        st.subheader("🗂 Proyectos / Contratos / Procesos")

        with st.form("form_cat_proyectos"):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("Nombre del proyecto / contrato / proceso", "")
                codigo = st.text_input("Código / identificador", "")
            with col2:
                descripcion = st.text_area("Descripción", "", height=80)

            submitted = st.form_submit_button("✅ Guardar proyecto en catálogo")

        if submitted:
            if nombre.strip() == "":
                st.warning("El nombre no puede estar vacío.")
            else:
                try:
                    insertar_en_catalogo("cat_proyectos", {
                        "nombre": nombre.strip(),
                        "codigo": codigo.strip(),
                        "descripcion": descripcion.strip()
                    })
                    st.success("✅ Proyecto/contrato/proceso agregado al catálogo.")
                except Exception as e:
                    st.error(f"No se pudo guardar el registro. Detalle: {e}")

        st.markdown("### 📋 Proyectos / Contratos / Procesos en catálogo")
        df_cat = cargar_catalogo("cat_proyectos")
        if df_cat.empty:
            st.info("Aún no hay registros en el catálogo.")
        else:
            st.dataframe(df_cat, width='stretch', height=300)

    # 3) Áreas
    with tabs[2]:
        st.subheader("🏢 Áreas")

        with st.form("form_cat_areas"):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("Nombre del área", "")
            with col2:
                descripcion = st.text_area("Descripción del área", "", height=80)

            submitted = st.form_submit_button("✅ Guardar área en catálogo")

        if submitted:
            if nombre.strip() == "":
                st.warning("El nombre del área no puede estar vacío.")
            else:
                try:
                    insertar_en_catalogo("cat_areas", {
                        "nombre": nombre.strip(),
                        "descripcion": descripcion.strip()
                    })
                    st.success("✅ Área agregada al catálogo.")
                except Exception as e:
                    st.error(f"No se pudo guardar el área. Detalle: {e}")

        st.markdown("### 📋 Áreas en catálogo")
        df_cat = cargar_catalogo("cat_areas")
        if df_cat.empty:
            st.info("Aún no hay áreas en el catálogo.")
        else:
            st.dataframe(df_cat, width='stretch', height=300)

    # 4) Responsables
    with tabs[3]:
        st.subheader("👤 Responsables")

        with st.form("form_cat_responsables"):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("Nombre del responsable", "")
                cargo = st.text_input("Cargo", "")
            with col2:
                correo = st.text_input("Correo electrónico", "")

            submitted = st.form_submit_button("✅ Guardar responsable en catálogo")

        if submitted:
            if nombre.strip() == "":
                st.warning("El nombre del responsable no puede estar vacío.")
            else:
                try:
                    insertar_en_catalogo("cat_responsables", {
                        "nombre": nombre.strip(),
                        "cargo": cargo.strip(),
                        "correo": correo.strip()
                    })
                    st.success("✅ Responsable agregado al catálogo.")
                except Exception as e:
                    st.error(f"No se pudo guardar el responsable. Detalle: {e}")

        st.markdown("### 📋 Responsables en catálogo")
        df_cat = cargar_catalogo("cat_responsables")
        if df_cat.empty:
            st.info("Aún no hay responsables en el catálogo.")
        else:
            st.dataframe(df_cat, width='stretch', height=300)

    # 5) Controles
    with tabs[4]:
        st.subheader("🛡 Controles")

        with st.form("form_cat_controles"):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("Nombre del control", "")
                tipo = st.text_input("Tipo de control (preventivo, detectivo, correctivo, etc.)", "")
            with col2:
                descripcion = st.text_area("Descripción del control", "", height=80)

            submitted = st.form_submit_button("✅ Guardar control en catálogo")

        if submitted:
            if nombre.strip() == "":
                st.warning("El nombre del control no puede estar vacío.")
            else:
                try:
                    insertar_en_catalogo("cat_controles", {
                        "nombre": nombre.strip(),
                        "descripcion": descripcion.strip(),
                        "tipo": tipo.strip()
                    })
                    st.success("✅ Control agregado al catálogo.")
                except Exception as e:
                    st.error(f"No se pudo guardar el control. Detalle: {e}")

        st.markdown("### 📋 Controles en catálogo")
        df_cat = cargar_catalogo("cat_controles")
        if df_cat.empty:
            st.info("Aún no hay controles en el catálogo.")
        else:
            st.dataframe(df_cat, width='stretch', height=300)
