import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from gtts import gTTS
import platform

# =========================
# Funciones adicionales
# =========================

# 🔹 Función 2: Extracción de palabras clave
def extraer_palabras_clave(texto):
    llm = OpenAI(model_name="gpt-4o", temperature=0)
    prompt = f"Extrae una lista de las 10 palabras o conceptos clave más importantes del siguiente texto:\n\n{texto[:4000]}"
    return llm(prompt)

# 🔹 Función 4: Convertir respuesta a audio
def respuesta_audio(texto):
    tts = gTTS(texto, lang="es")
    tts.save("respuesta.mp3")
    audio_file = open("respuesta.mp3", "rb")
    st.audio(audio_file.read(), format="audio/mp3")

# 🔹 Función 5: Explicación tipo profesor
def explicacion_profesor(texto):
    llm = OpenAI(model_name="gpt-4o", temperature=0.5)
    prompt = f"Explica el siguiente texto de manera sencilla y educativa, como si fueras un profesor explicándole a un estudiante:\n\n{texto[:4000]}"
    return llm(prompt)

# =========================
# Interfaz principal
# =========================

st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Cargar imagen
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar
with st.sidebar:
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")
    st.info("Nuevas funciones: Extracción de palabras clave, modo profesor y respuesta en audio")

# Clave API
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Subida de PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Procesamiento
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.info(f"Texto extraído: {len(text)} caracteres")

        # Dividir texto
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")

        # Embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        # =========================
        # Función principal de RAG
        # =========================
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.markdown("### 💬 Respuesta:")
            st.markdown(response)

            # --- Botón para generar audio de la respuesta ---
            if st.button("🔊 Escuchar respuesta"):
                with st.spinner("Generando audio..."):
                    respuesta_audio(response)
                    st.success("Audio generado correctamente 🎧")

        # =========================
        # Función 2: Palabras clave
        # =========================
        st.markdown("---")
        st.subheader("📚 Análisis del documento")

        if st.button("Extraer palabras clave"):
            with st.spinner("Analizando texto..."):
                palabras = extraer_palabras_clave(text)
                st.markdown("### 🔑 Palabras clave:")
                st.write(palabras)

        # =========================
        # Función 5: Explicación tipo profesor
        # =========================
        if st.button("👨‍🏫 Explicación tipo profesor"):
            with st.spinner("Generando explicación educativa..."):
                explicacion = explicacion_profesor(text)
                st.markdown("### 👨‍🏫 Explicación del documento:")
                st.write(explicacion)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
