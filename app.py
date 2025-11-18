import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
import sqlite3
import hashlib
import io
import traceback
from typing import Optional, Tuple

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
    """
    Conexión a SQLite. check_same_thread=False para evitar errores con Streamlit y caching.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


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

        conn.commit()


@st.cache_data
def cargar_bd() -> pd.DataFrame:
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
        # Si falla la lectura, devolver DF vacío con columnas
        return pd.DataFrame(columns=columnas)


def limpiar_cache_bd():
    cargar_bd.clear()


def insertar_df_en_bd(df_bd: pd.DataFrame):
    """Inserta un DataFrame con las columnas de la matriz en la tabla riesgos."""
    if df_bd is None or df_bd.empty:
        return
    try:
        with get_connection() as conn:
            df_bd.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
            conn.commit()
        limpiar_cache_bd()
    except Exception as e:
        st.error("Error al insertar registros en la base de datos.")
        st.exception(e)


def cargar_planes() -> pd.DataFrame:
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
    if df_plan is None or df_plan.empty:
        return
    try:
        with get_connection() as conn:
            df_plan.to_sql("planes_tratamiento", conn, if_exists="append", index=False)
            conn.commit()
    except Exception as e:
        st.error("Error al insertar plan de tratamiento.")
        st.exception(e)


# ======================
# FUNCIONES AUXILIARES DE RIESGO
# ======================
def get_column(df: pd.DataFrame, target_name: str) -> Optional[str]:
    """Devuelve el nombre real de la columna en el DataFrame, buscando por coincidencia en minúsculas."""
    cols_lower = {c.lower(): c for c in df.columns}
    return cols_lower.get(target_name.lower())


def calcular_nivel_riesgo(df: pd.DataFrame, col_prob: str, col_cons: str, col_nivel: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
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
    Puedes ajustar estos rangos según tu metodología.
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


def detectar_columna_fecha(df: pd.DataFrame) -> Tuple[Optional[str], pd.DataFrame]:
    """Intenta detectar automáticamente una columna de fecha."""
    # Preferencias por nombres
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ["fecha", "fecha evento", "date", "fecha_riesgo", "fecha de riesgo"]:
        if cand in cols_lower:
            col = cols_lower[cand]
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                if df[col].notna().any():
                    return col, df
            except Exception:
                pass

    # Buscar cualquier columna que contenga la palabra 'fecha'
    for c in df.columns:
        if "fecha" in c.lower():
            try:
                converted = pd.to_datetime(df[c], errors="coerce")
                if converted.notna().sum() > 0:
                    df[c] = converted
                    return c, df
            except Exception:
                continue

    # Columnas que ya son datetime
    datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols:
        return datetime_cols[0], df

    # Intentar convertir columnas objeto
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


def detectar_fila_encabezado(df_raw: pd.DataFrame, nombres_minimos, max_filas_busqueda=15) -> Optional[int]:
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


def calcular_kpis_y_graficos(df: pd.DataFrame, col_area: str = "Área", titulo_prefix: str = ""):
    """KPIs + gráficos + tabla a partir de un DataFrame de riesgos."""
    if df is None or df.empty:
        st.info("No hay datos para mostrar aún.")
        return

    # KPIs
    total_riesgos = len(df)
    total_alto = int((df["Categoría Riesgo"] == "Alto").sum())
    total_medio = int((df["Categoría Riesgo"] == "Medio").sum())
    total_bajo = int((df["Categoría Riesgo"] == "Bajo").sum())

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
                <div style="font-size:0.9rem;font-weight:600;color:#ffffff">Riesgos Altos</div>
                <div style="font-size:1.8rem;font-weight:800;color:#ffffff">{total_alto}</div>
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
            st.plotly_chart(fig_bar, use_container_width=True)
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
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("Sin datos de categoría para mostrar.")

    # 🔥 Matriz de calor Probabilidad x Consecuencia
    st.subheader(f"🧱 {titulo_prefix}Matriz de calor Probabilidad x Consecuencia")
    if "Probabilidad" in df.columns and "Consecuencia" in df.columns:
        df_pc = df.copy()
        df_pc["Probabilidad"] = pd.to_numeric(df_pc["Probabilidad"], errors="coerce")
        df_pc["Consecuencia"] = pd.to_numeric(df_pc["Consecuencia"], errors="coerce")
        df_pc = df_pc.dropna(subset=["Probabilidad", "Consecuencia"])

        if df_pc.empty:
            st.info("No hay datos suficientes para construir la matriz de calor.")
        else:
            tabla = (
                df_pc
                .groupby(["Probabilidad", "Consecuencia"])
                .size()
                .reset_index(name="Total")
            )
            matriz = tabla.pivot(
                index="Probabilidad",
                columns="Consecuencia",
                values="Total"
            ).fillna(0)

            matriz = matriz.sort_index().sort_index(axis=1)

            fig_heat = px.imshow(
                matriz,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdYLGn",
                labels={"color": "N° riesgos"},
                title=f"{titulo_prefix}Matriz de calor P x C"
            )
            fig_heat.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#000000",
                xaxis_title="Consecuencia",
                yaxis_title="Probabilidad"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
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
                st.plotly_chart(fig_line, use_container_width=True)
        except Exception:
            st.info("No fue posible procesar la columna Fecha para tendencias.")
    else:
        st.info("No se encontró columna de fecha para mostrar tendencias.")

    st.markdown("---")

    # Tabla
    st.subheader(f"📋 {titulo_prefix}Tabla de riesgos")
    st.dataframe(df, width='content', height=400)


def integrar_archivo_a_bd(file_bytes: bytes, filename: str):
    """
    Lee un archivo (Excel/CSV), detecta encabezados, normaliza columnas
    y lo integra como registros en la base de datos global.
    Acepta encabezado 'Riesgo' o 'Peligro', pero lo mapea a 'Riesgo'.
    Muestra mensajes claros al usuario y previene duplicados por hash.
    """
    try:
        file_hash = hashlib.md5(file_bytes).hexdigest()

        # Verificar si ese hash ya existe en la BD
        df_bd = cargar_bd()
        if "FileHash" in df_bd.columns and not df_bd.empty:
            if file_hash in df_bd["FileHash"].dropna().unique().tolist():
                st.info("Este archivo ya fue integrado previamente a la base de datos global.")
                return

        # Leer sin encabezados
        if filename.lower().endswith(".csv"):
            df_raw = pd.read_csv(io.BytesIO(file_bytes), header=None, dtype=str)
        else:
            df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None, dtype=str)

        nombres_minimos = ["Riesgo", "Peligro", "Probabilidad", "Consecuencia", "Área", "Area", "Responsable", "Control"]
        fila_header = detectar_fila_encabezado(df_raw, nombres_minimos)

        if fila_header is None:
            st.error(
                "❌ No se pudo detectar la fila de encabezados. "
                "Verifica que tu archivo tenga columnas como `Riesgo` (o `Peligro`), "
                "`Probabilidad`, `Consecuencia`, `Área`, `Responsable`, `Control` en alguna de las primeras filas."
            )
            st.write("Primeras filas detectadas:")
            st.dataframe(df_raw.head(10), width=None)
            return

        header_row = df_raw.iloc[fila_header].astype(str).str.strip()
        df = df_raw.iloc[fila_header + 1:].copy()
        df.columns = header_row
        df = df.reset_index(drop=True)
        df.columns = df.columns.map(lambda x: str(x).strip())

        # Identificar columnas clave (insensible a mayúsculas/tildes)
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

        fecha_vals = None
        if col_fecha is not None:
            try:
                fecha_vals = pd.to_datetime(df[col_fecha], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                fecha_vals = None

        if col_obs is not None:
            obs_vals = df[col_obs].fillna("").astype(str)
        else:
            obs_vals = ""

        proy_vals = df[col_proy] if col_proy is not None else ""
        estado_vals = df[col_estado] if col_estado is not None else "Identificado"
        criticidad_vals = df[col_criticidad] if col_criticidad is not None else ""

        df_para_bd = pd.DataFrame({
            "Riesgo": df[col_riesgo].fillna("").astype(str),
            "Probabilidad": df[col_prob],
            "Consecuencia": df[col_cons],
            "Nivel de Riesgo": df[col_nivel],
            "Categoría Riesgo": df["Categoría Riesgo"],
            "Área": df[col_area].fillna("").astype(str),
            "Responsable": df[col_resp].fillna("").astype(str),
            "Control": df[col_control].fillna("").astype(str),
            "Fecha": fecha_vals,
            "Observaciones": obs_vals,
            "Origen": "Archivo",
            "FileHash": file_hash,
            "Proyecto": proy_vals.fillna("").astype(str) if hasattr(proy_vals, "fillna") else proy_vals,
            "Estado": estado_vals.fillna("").astype(str) if hasattr(estado_vals, "fillna") else estado_vals,
            "Criticidad": criticidad_vals.fillna("").astype(str) if hasattr(criticidad_vals, "fillna") else criticidad_vals
        })

        insertar_df_en_bd(df_para_bd)
        st.success(f"✅ Archivo integrado exitosamente a la base de datos global de riesgos. Registros añadidos: {len(df_para_bd)}")
    except Exception as e:
        st.error("Ocurrió un error al procesar el archivo.")
        st.exception(traceback.format_exc())


# ======================
# INICIO APP
# ======================
init_db()

with st.sidebar:
    st.markdown("## ⚙️ Navegación")
    vista = st.radio(
        "Selecciona la vista:",
        ["Dashboard de la base de datos",
         "Formulario en tiempo real",
         "Gestión y actualización",
         "Planes de tratamiento"],
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

    # --- Carga de archivo e integración a la BD global (no visualización directa) ---
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

    # --- SIEMPRE trabajamos sobre la base de datos global ---
    bd_global = cargar_bd()

    with st.sidebar:
        st.markdown("---")
        st.header("🎛️ Filtros sobre la base de datos global")
        if bd_global.empty:
            st.info("Aún no hay datos en la base de datos global.")
            filter_area = filter_resp = filter_cat = filter_origen = []
            search_text = ""
        else:
            areas = sorted([a for a in bd_global["Área"].dropna().unique().tolist() if a != ""])
            responsables = sorted([r for r in bd_global["Responsable"].dropna().unique().tolist() if r != ""])
            categorias = sorted([c for c in bd_global["Categoría Riesgo"].dropna().unique().tolist() if c != ""])
            origenes = sorted([o for o in bd_global["Origen"].dropna().unique().tolist() if o != ""])

            filter_area = st.multiselect("Filtrar por Área", options=areas, default=areas)
            filter_resp = st.multiselect("Filtrar por Responsable", options=responsables, default=responsables)
            filter_cat = st.multiselect("Filtrar por Categoría de Riesgo", options=categorias, default=categorias)
            filter_origen = st.multiselect("Filtrar por Origen", options=origenes, default=origenes)
            search_text = st.text_input("🔍 Buscar (Riesgo / Control)", "")

    st.markdown("### 📚 Base de datos global de riesgos")

    if bd_global.empty:
        st.info("No hay registros aún. Carga un archivo o registra riesgos en el formulario.")
    else:
        # Partimos SIEMPRE de la BD global
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

        # 🎯 DASHBOARD sobre df_filtrado (BD global filtrada)
        calcular_kpis_y_graficos(df_filtrado, col_area="Área", titulo_prefix="BD Global - ")

        # --- Export general ---
        st.markdown("### 📤 Exportar datos filtrados para Power BI / Excel")
        csv_data = df_filtrado.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="💾 Descargar CSV filtrado",
            data=csv_data,
            file_name="matriz_riesgos_bd_filtrada.csv",
            mime="text/csv"
        )

        # --- Export directo de Riesgos Altos ---
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
            label="💾 Descargar BD completa (CSV)",
            data=csv_bd,
            file_name="matriz_riesgos_bd_completa.csv",
            mime="text/csv"
        )

        # Export directo de Riesgos Altos (sobre la BD completa)
        st.markdown("### 🚨 Exportar solo Riesgos Altos")
        df_altos = bd[bd["Categoría Riesgo"] == "Alto"].copy()

        if df_altos.empty:
            st.info("No hay riesgos altos en la base de datos.")
        else:
            csv_altos = df_altos.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="💾 Descargar solo Riesgos Altos (CSV)",
                data=csv_altos,
                file_name="matriz_riesgos_altos_bd.csv",
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
                try:
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
                except Exception as e:
                    st.error("Error al actualizar el riesgo.")
                    st.exception(e)

# ----------------------------------------
# VISTA 4: PLANES DE TRATAMIENTO
# ----------------------------------------
else:  # vista == "Planes de tratamiento"
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
                st.dataframe(df_planes_riesgo, width=None, height=300)
            df_planes = cargar_planes()
            df_planes_riesgo = df_planes[df_planes["riesgo_id"] == int(riesgo_id)]

            if df_planes_riesgo.empty:
                st.info("Este riesgo aún no tiene planes de tratamiento registrados.")
            else:
                st.dataframe(df_planes_riesgo, width='stretch', height=300)

