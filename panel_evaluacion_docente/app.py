import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from pysentimiento import create_analyzer
import nltk
from nltk.corpus import stopwords

# Descargar stopwords de forma silenciosa para el servidor
nltk.download('stopwords', quiet=True)

# ---------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------
st.set_page_config(page_title="Evaluación Docente UDLA", layout="wide")
st.title("📊 Sistema Inteligente de Evaluación Docente (NLP)")
st.markdown("Prototipo basado en CRISP-ML(Q) para análisis cuantitativo y cualitativo.")

# ---------------------------------------------------------
# CACHÉ DEL MODELO (Para que la app no sea lenta)
# ---------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    return create_analyzer(task="sentiment", lang="es")

analyzer = cargar_modelo()

# ---------------------------------------------------------
# FUNCIONES DE PROCESAMIENTO
# ---------------------------------------------------------
def limpiar_texto(texto):
    if pd.isna(texto): return ""
    texto = str(texto).lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)
    return ' '.join(texto.split())

def procesar_sentimiento(texto):
    try:
        res = analyzer.predict(texto)
        return res.output, res.probas[res.output]
    except:
        return "NEU", 0.0

# ---------------------------------------------------------
# INTERFAZ DE CARGA (BARRA LATERAL)
# ---------------------------------------------------------
st.sidebar.header("Carga de Datos")
archivo_subido = st.sidebar.file_uploader("Sube el Informe de Resumen (CSV)", type=['csv'])

if archivo_subido is not None:
    # 1. Obtener el nombre del archivo físico
    nombre_archivo_completo = archivo_subido.name
    
    # 2. Lógica de extracción: Dividimos el nombre por el guion " - "
    # Ejemplo: "Evaluación... - 118290304 - Rossana Mendoza - Informe..."
    partes = nombre_archivo_completo.split(" - ")
    
    if len(partes) >= 3:
        # El nombre del docente está en la tercera posición (índice 2)
        nombre_docente = partes[2]
    else:
        nombre_docente = "Docente No Identificado"

    df = pd.read_csv(archivo_subido)
    st.sidebar.success(f"Archivo de {nombre_docente} cargado.")

    if st.button("Ejecutar Análisis Completo"):
        with st.spinner(f'Analizando el desempeño de {nombre_docente}...'):
            
            # (Aquí iría el resto de tu lógica de procesamiento...)
            
            # --- MOSTRAR RESULTADOS ---
            st.markdown("---")
            # Mostramos el nombre extraído en el título principal
            st.header(f"Reporte de Evaluación: {nombre_docente}")
            
            st.subheader("1. Evaluación Cuantitativa")
          
            # --- 1. CÁLCULO CUANTITATIVO (Nota 1-5) ---
            mapeo_puntos = {
                'Totalmente en desacuerdo': 1, 'En desacuerdo': 2,
                'Neutral': 3, 'De acuerdo': 4, 'Totalmente de acuerdo': 5
            }
            df_likert = df[df['Q Type'] == 'LIK'].copy()
            df_likert['Puntos'] = df_likert['Answer Match'].map(mapeo_puntos)
            df_likert['Puntaje_Total'] = df_likert['Puntos'] * df_likert['# Responses']
            total_votos = df_likert['# Responses'].sum()
            nota_final = (df_likert['Puntaje_Total'].sum() / total_votos) if total_votos > 0 else 0

            # --- 2. PROCESAMIENTO NLP (Limpieza y Clustering) ---
            df_comentarios = df[df['Q Type'] == 'RE'].copy()
            df_comentarios['Comentario_Limpio'] = df_comentarios['Answer'].apply(limpiar_texto)
            df_comentarios = df_comentarios[df_comentarios['Comentario_Limpio'].str.len() > 3].reset_index(drop=True)
            
            stop_words_es = stopwords.words('spanish')
            vectorizador = TfidfVectorizer(stop_words=stop_words_es, max_df=0.85, min_df=2)
            
            # Solo hace clustering si hay suficientes comentarios
            if len(df_comentarios) >= 3:
                X_tfidf = vectorizador.fit_transform(df_comentarios['Comentario_Limpio'])
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                df_comentarios['Tema (Cluster)'] = kmeans.fit_predict(X_tfidf)
            else:
                df_comentarios['Tema (Cluster)'] = 0

            # --- 3. ANÁLISIS DE SENTIMIENTO (RoBERTa) ---
            resultados_ia = df_comentarios['Comentario_Limpio'].apply(procesar_sentimiento)
            df_comentarios['Sentimiento'] = [r[0] for r in resultados_ia]
            df_comentarios['Confianza'] = [r[1] for r in resultados_ia]
            df_comentarios['Clasificación'] = df_comentarios['Sentimiento'].map({'POS': 'Positivo', 'NEG': 'Negativo', 'NEU': 'Neutro'})

            # ---------------------------------------------------------
            # VISUALIZACIÓN DEL DASHBOARD
            # ---------------------------------------------------------
            st.markdown("---")
            st.subheader("1. Evaluación Cuantitativa")
            
            # Mostrar la nota general en grande
            st.metric(label="Nota General del Docente", value=f"{nota_final:.2f} / 5.00", delta=f"{total_votos} respuestas analizadas", delta_color="normal")

            st.markdown("---")
            st.subheader("2. Análisis Cualitativo y MLOps")
            
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Distribución de Sentimientos por Tema**")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df_comentarios, x='Tema (Cluster)', hue='Clasificación', 
                              palette={'Positivo': '#2ecc71', 'Negativo': '#e74c3c', 'Neutro': '#95a5a6'}, ax=ax)
                st.pyplot(fig)

            with col2:
                st.markdown("**Métricas de Quality Gate (Confianza del Modelo)**")
                confianza_media = df_comentarios['Confianza'].mean() * 100
                baja_confianza = len(df_comentarios[df_comentarios['Confianza'] < 0.60])
                
                st.info(f"**Certeza Promedio de la IA:** {confianza_media:.2f}%")
                if baja_confianza > 0:
                    st.warning(f"⚠️ {baja_confianza} comentarios tienen una certeza menor al 60% y requieren revisión humana.")
                else:
                    st.success("✅ Todos los comentarios fueron clasificados con alta certeza. No se requiere revisión manual.")

            st.markdown("---")
            st.subheader("Detalle de Comentarios")
            st.dataframe(df_comentarios[['Answer', 'Tema (Cluster)', 'Clasificación', 'Confianza']])

else:
    st.info("👈 Sube tu archivo CSV en la barra lateral para comenzar.")