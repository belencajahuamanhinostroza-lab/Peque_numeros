
import os
import numpy as np
import streamlit as st
import tensorflow as tf

from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="PequeNúmeros | Aprende jugando",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# ESTADO DE LA APLICACIÓN
# ==========================================================

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if "historial" not in st.session_state:
    st.session_state.historial = []

if "puntos" not in st.session_state:
    st.session_state.puntos = 0


# ==========================================================
# ESTILOS
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

.stApp {
    background: #FFF9F5;
}

header {
    visibility: hidden;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ENCABEZADO */

.hero {
    background: linear-gradient(120deg, #DCCEFF, #FCE1EA);
    border-radius: 28px;
    padding: 35px;
    margin-bottom: 25px;
}

.logo {
    color: #7552B5;
    font-size: 17px;
    font-weight: 900;
    letter-spacing: 1px;
}

.hero-title {
    color: #49376C;
    font-size: 38px;
    font-weight: 900;
    line-height: 1.2;
    margin-top: 15px;
}

.hero-description {
    color: #685C7E;
    font-size: 16px;
    margin-top: 12px;
    line-height: 1.7;
}

/* TÍTULOS */

.section-title {
    font-size: 25px;
    font-weight: 900;
    color: #49376C;
    margin-bottom: 5px;
}

.section-subtitle {
    font-size: 14px;
    color: #897F96;
    margin-bottom: 20px;
}

/* TARJETAS */

.card {
    background: white;
    border: 2px solid #F1E8F5;
    border-radius: 20px;
    padding: 20px;
    height: 100%;
}

.number-card {
    background: #FFFFFF;
    border: 2px solid #F0E6F5;
    border-radius: 20px;
    padding: 15px;
    text-align: center;
}

.number-big {
    font-size: 65px;
    font-weight: 900;
    color: #8A69CA;
}

.number-name {
    font-size: 17px;
    font-weight: 800;
    color: #57456F;
}

/* RESULTADO */

.result-card {
    background: linear-gradient(145deg, #E9DEFF, #FCE4ED);
    border-radius: 25px;
    padding: 25px;
    text-align: center;
}

.result-number {
    font-size: 110px;
    font-weight: 900;
    color: #7653B8;
    line-height: 1.2;
}

.result-label {
    color: #68527F;
    font-weight: 800;
    font-size: 16px;
}

.confidence {
    font-size: 30px;
    font-weight: 900;
    color: #8B62C5;
}

/* PROGRESO */

.points-card {
    background: #FFF0D9;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
}

.points-number {
    color: #B77925;
    font-size: 30px;
    font-weight: 900;
}

/* BOTONES */

.stButton > button {
    border-radius: 15px;
    min-height: 48px;
    font-weight: 800;
    border: 2px solid #E9DDF4;
    color: #70549C;
    background: white;
}

.stButton > button[kind="primary"] {
    background: #8968C5;
    border: none;
    color: white;
}

.stButton > button[kind="primary"]:hover {
    background: #7250B0;
    color: white;
}

/* PESTAÑAS */

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: #F4EAF8;
    border-radius: 12px;
    padding: 12px 20px;
    color: #725B91;
    font-weight: 800;
}

.stTabs [aria-selected="true"] {
    background: #E6D8FA !important;
    color: #6947A7 !important;
}

.footer {
    text-align: center;
    color: #A69BAF;
    font-size: 12px;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# NOMBRES DE LOS NÚMEROS
# ==========================================================

numeros = {
    0: "Cero",
    1: "Uno",
    2: "Dos",
    3: "Tres",
    4: "Cuatro",
    5: "Cinco",
    6: "Seis",
    7: "Siete",
    8: "Ocho",
    9: "Nueve"
}

emojis = {
    0: "🍩",
    1: "🦋",
    2: "🐥",
    3: "🍓",
    4: "⭐",
    5: "🌸",
    6: "🐢",
    7: "🌈",
    8: "🐙",
    9: "🍉"
}


# ==========================================================
# CARGAR MODELO
# ==========================================================

@st.cache_resource
def cargar_modelo():

    ruta = "mnist_model.h5"

    if not os.path.exists(ruta):
        return None

    return keras.models.load_model(ruta)


try:
    modelo = cargar_modelo()

except Exception as error:
    modelo = None
    error_modelo = str(error)


# ==========================================================
# PREPROCESAMIENTO DE IMAGEN
# ==========================================================

def procesar_imagen(imagen_canvas):

    # Convertir el dibujo a escala de grises
    imagen = Image.fromarray(
        imagen_canvas.astype("uint8")
    ).convert("L")

    matriz = np.array(imagen)

    # Detectar los píxeles blancos del dibujo
    mascara = matriz > 30

    coordenadas = np.argwhere(mascara)

    if len(coordenadas) == 0:
        return None

    # Encontrar los límites del número
    y_min, x_min = coordenadas.min(axis=0)
    y_max, x_max = coordenadas.max(axis=0)

    recorte = imagen.crop((
        x_min,
        y_min,
        x_max + 1,
        y_max + 1
    ))

    # Mantener proporciones
    recorte.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Crear imagen MNIST de 28x28
    imagen_final = Image.new(
        "L",
        (28, 28),
        0
    )

    # Centrar el número
    x = (28 - recorte.width) // 2
    y = (28 - recorte.height) // 2

    imagen_final.paste(
        recorte,
        (x, y)
    )

    return imagen_final


# ==========================================================
# ENCABEZADO PRINCIPAL
# ==========================================================

st.markdown("""
<div class="hero">

    <div class="logo">
        🌈 PEQUE NÚMEROS
    </div>

    <div class="hero-title">
        ¡Hola, pequeño explorador! 👋
    </div>

    <div class="hero-description">
        ¿Estás listo para descubrir el maravilloso mundo
        de los números? Aprende, dibuja y juega con nosotros.
        <br>
        <b>¡Dibuja, descubre y aprende!</b>
    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================================
# COMPROBAR MODELO
# ==========================================================

if modelo is None:

    st.error(
        "No se pudo cargar mnist_model.h5. "
        "Comprueba que el archivo esté en la carpeta del proyecto."
    )

    if "error_modelo" in globals():
        st.code(error_modelo)

    st.stop()


# ==========================================================
# INFORMACIÓN DEL JUGADOR
# ==========================================================

col1, col2 = st.columns([2, 1])

with col1:

    st.markdown("""
    <div class="section-title">
        ¡Vamos a aprender! 🎨
    </div>
    <div class="section-subtitle">
        Explora las actividades y conviértete en un experto
        de los números.
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="points-card">
        <div style="font-size:14px;color:#9B722E;font-weight:800;">
            ⭐ MIS ESTRELLITAS
        </div>
        <div class="points-number">
            {st.session_state.puntos} ⭐
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# NAVEGACIÓN
# ==========================================================

tab1, tab2, tab3 = st.tabs([
    "🏠 Aprende",
    "✏️ Juega y dibuja",
    "🏆 Mis logros"
])


# ==========================================================
# PESTAÑA 1: APRENDE
# ==========================================================

with tab1:

    st.markdown("""
    <div class="section-title">
        Conoce los números 🔢
    </div>

    <div class="section-subtitle">
        ¡Cada número tiene un nombre! Selecciona uno para conocerlo.
    </div>
    """, unsafe_allow_html=True)

    # Mostrar números del 0 al 9
    columnas = st.columns(5)

    for i in range(10):

        with columnas[i % 5]:

            st.markdown(f"""
            <div class="number-card">
                <div class="number-big">{i}</div>
                <div class="number-name">{numeros[i]}</div>
                <div style="font-size:30px;margin-top:8px;">
                    {emojis[i]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

    st.divider()

    st.markdown("""
    <div class="section-title">
        ¡Elige un número para aprender! 💜
    </div>
    """, unsafe_allow_html=True)

    numero_elegido = st.selectbox(
        "Selecciona un número:",
        list(range(10)),
        format_func=lambda x: f"{x} - {numeros[x]}"
    )

    st.markdown(f"""
    <div class="result-card">

        <div style="font-size:18px;color:#765B9A;font-weight:800;">
            ¡Este es el número!
        </div>

        <div class="result-number">
            {numero_elegido}
        </div>

        <div class="result-label">
            {numeros[numero_elegido].upper()}
        </div>

        <div style="font-size:55px;margin-top:10px;">
            {emojis[numero_elegido]}
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.info(
        f"💡 Recuerda: el número {numero_elegido} "
        f"se llama {numeros[numero_elegido]}."
    )


# ==========================================================
# PESTAÑA 2: JUEGA Y DIBUJA
# ==========================================================

with tab2:

    st.markdown("""
    <div class="section-title">
        ¡Es hora de dibujar! ✏️
    </div>

    <div class="section-subtitle">
        Dibuja un número del 0 al 9 y nuestra inteligencia
        artificial intentará descubrir cuál es.
    </div>
    """, unsafe_allow_html=True)

    col_canvas, col_info = st.columns(
        [1.5, 1],
        gap="large"
    )

    with col_canvas:

        st.markdown("**🎨 Tu espacio de dibujo**")

        canvas_result = st_canvas(

            fill_color="rgba(255,255,255,0)",

            stroke_width=18,

            stroke_color="#FFFFFF",

            background_color="#17152B",

            width=450,

            height=350,

            drawing_mode="freedraw",

            key=f"canvas_{st.session_state.canvas_version}",

            display_toolbar=True

        )

        st.caption(
            "Usa el mouse o tu dedo para dibujar un número."
        )

        boton_predecir = st.button(
            "✨ ¡Descubrir mi número!",
            type="primary",
            use_container_width=True
        )

        boton_limpiar = st.button(
            "🧹 Borrar y volver a dibujar",
            use_container_width=True
        )

        if boton_limpiar:

            st.session_state.canvas_version += 1
            st.session_state.resultado = None

            st.rerun()

    with col_info:

        st.markdown("""
        <div class="card">

            <div style="font-size:45px;text-align:center;">
                🧸
            </div>

            <div style="font-size:21px;font-weight:900;
            color:#654A91;text-align:center;">
                ¡Hola, amiguito!
            </div>

            <p style="color:#857993;text-align:center;">
                Escribe un número y yo intentaré
                adivinar cuál dibujaste.
            </p>

            <hr style="border:1px solid #F0E8F5;">

            <div style="font-weight:800;color:#654A91;">
                ¿Cómo jugar?
            </div>

            <p style="color:#857993;">
                1. Dibuja un número.<br>
                2. Presiona descubrir.<br>
                3. Mira qué número reconocí.<br>
                4. ¡Sigue practicando!
            </p>

        </div>
        """, unsafe_allow_html=True)


    # ------------------------------------------------------
    # PREDICCIÓN
    # ------------------------------------------------------

    if boton_predecir:

        if canvas_result.image_data is None:

            st.warning("¡Primero dibuja un número!")

        else:

            imagen_canvas = canvas_result.image_data

            imagen_procesada = procesar_imagen(
                imagen_canvas
            )

            if imagen_procesada is None:

                st.warning(
                    "¡Tu hoja está vacía! Dibuja un número."
                )

            else:

                with st.spinner(
                    "🔍 Estoy descubriendo tu número..."
                ):

                    matriz = np.array(
                        imagen_procesada,
                        dtype=np.float32
                    ) / 255.0

                    entrada = matriz.reshape(
                        1, 28, 28, 1
                    )

                    prediccion = modelo.predict(
                        entrada,
                        verbose=0
                    )[0]

                    # Obtener número reconocido
                    digito = int(
                        np.argmax(prediccion)
                    )

                    confianza = float(
                        np.max(prediccion) * 100
                    )

                    # Segunda posibilidad
                    indices = np.argsort(prediccion)

                    alternativa = int(
                        indices[-2]
                    )

                    confianza_alt = float(
                        prediccion[alternativa] * 100
                    )

                    # Guardar resultado
                    st.session_state.resultado = {

                        "digito": digito,

                        "confianza": confianza,

                        "alternativa": alternativa,

                        "confianza_alt": confianza_alt,

                        "imagen_original": imagen_canvas,

                        "imagen_procesada": imagen_procesada,

                        "probabilidades": prediccion.tolist()

                    }

                    # Guardar en historial
                    st.session_state.historial.append({
                        "numero": digito,
                        "nombre": numeros[digito],
                        "confianza": confianza
                    })

                    # Otorgar estrella si la confianza supera 70%
                    if confianza >= 70:
                        st.session_state.puntos += 1

                st.rerun()


    # ------------------------------------------------------
    # MOSTRAR RESULTADO
    # ------------------------------------------------------

    if st.session_state.resultado is not None:

        resultado = st.session_state.resultado

        st.write("")
        st.divider()

        st.markdown("""
        <div class="section-title">
            ¡Mira lo que descubrí! 🎉
        </div>
        """, unsafe_allow_html=True)

        if resultado["confianza"] >= 70:

            st.balloons()

            mensaje = "¡Excelente! ¡La IA reconoció tu número!"

        else:

            mensaje = (
                "¡Buen intento! Prueba dibujándolo "
                "un poquito más grande y claro."
            )

        st.markdown(f"""
        <div class="result-card">

            <div style="font-size:19px;color:#765B9A;font-weight:900;">
                {mensaje}
            </div>

            <div class="result-number">
                {resultado["digito"]}
            </div>

            <div class="result-label">
                {numeros[resultado["digito"]]}
                {emojis[resultado["digito"]]}
            </div>

            <div style="margin-top:15px;">
                <div class="result-label">
                    Seguridad estimada del modelo
                </div>

                <div class="confidence">
                    {resultado["confianza"]:.1f}%
                </div>
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Número reconocido",
                resultado["digito"]
            )

        with c2:

            st.metric(
                "Segunda posibilidad",
                resultado["alternativa"],
                f'{resultado["confianza_alt"]:.1f}%'
            )

        st.write("")

        st.markdown("**🖼️ Así procesó tu dibujo la IA**")

        img1, img2 = st.columns(2)

        with img1:

            st.caption("Dibujo original")

            st.image(
                resultado["imagen_original"],
                use_container_width=True
            )

        with img2:

            st.caption("Imagen preparada para MNIST")

            st.image(
                resultado["imagen_procesada"],
                width=180
            )

        st.write("")

        with st.expander("🔍 Ver cómo reconoció los números"):

            probabilidades = resultado["probabilidades"]

            for i, prob in enumerate(probabilidades):

                porcentaje = float(prob) * 100

                st.write(
                    f"**{i} - {numeros[i]}:** "
                    f"{porcentaje:.2f}%"
                )

                st.progress(
                    float(np.clip(prob, 0, 1))
                )


# ==========================================================
# PESTAÑA 3: MIS LOGROS
# ==========================================================

with tab3:

    st.markdown("""
    <div class="section-title">
        ¡Mi colección de números! 🏆
    </div>

    <div class="section-subtitle">
        Aquí puedes consultar los números que has dibujado.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="points-card">
        <div style="font-size:18px;font-weight:800;color:#9B722E;">
            ⭐ Estrellitas conseguidas
        </div>
        <div class="points-number">
            {st.session_state.puntos}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    numeros_encontrados = sorted(set(
        item["numero"]
        for item in st.session_state.historial
    ))

    st.markdown("**Números que has descubierto:**")

    columnas = st.columns(5)

    for i in range(10):

        with columnas[i % 5]:

            if i in numeros_encontrados:

                st.markdown(f"""
                <div class="number-card"
                style="background:#E9F8EB;border-color:#B8E7C0;">
                    <div class="number-big"
                    style="color:#469D62;">{i}</div>
                    <div class="number-name">{numeros[i]}</div>
                    <div style="font-size:22px;">✅</div>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="number-card">
                    <div class="number-big"
                    style="color:#C9C1D2;">{i}</div>
                    <div class="number-name"
                    style="color:#AAA0B4;">{numeros[i]}</div>
                    <div style="font-size:22px;">🔒</div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")

    st.divider()

    st.markdown("**📚 Mi historial de juegos**")

    if st.session_state.historial:

        for i, item in enumerate(
            reversed(st.session_state.historial),
            start=1
        ):

            st.write(
                f"**Partida {i}:** "
                f"{item['numero']} - {item['nombre']} "
                f"({item['confianza']:.1f}%)"
            )

    else:

        st.info(
            "Todavía no tienes números descubiertos. "
            "¡Ve a jugar y empieza tu colección!"
        )


# ==========================================================
# PIE DE PÁGINA
# ==========================================================

st.markdown("""
<div class="footer">

    🌈 <b>PequeNúmeros</b>

    <br>

    ¡Dibuja, descubre y aprende!

    <br><br>

    Aplicación educativa desarrollada con
    Python · Streamlit · TensorFlow · MNIST

</div>
""", unsafe_allow_html=True)
