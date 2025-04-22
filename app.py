import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
import base64
import google.generativeai as genai
from gtts import gTTS
from pathlib import Path

# --- Page Setup ---
st.set_page_config(page_title="📄🔊 PDF to Audio Summary", layout="centered")
st.title("📄✨ PDF Audio Summarizer using Gemini + TTS")

# --- Sidebar for Gemini API Key ---
st.sidebar.title("🔐 Gemini API Key")
api_key = st.sidebar.text_input("Enter your Gemini API key:", type="password")

# --- Function to extract text from PDF ---
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    text = ""
    with fitz.open(tmp_file_path) as doc:
        for page in doc:
            text += page.get_text()

    os.remove(tmp_file_path)
    return text.strip()

# --- Function to generate summary using Gemini ---
def generate_summary(text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = "Summarize this PDF content in a friendly voiceover style overview:\n\n" + text
    response = model.generate_content(prompt)
    return response.text

# --- Function to convert text to speech ---
def text_to_audio(text, filename="summary.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# --- Upload PDF ---
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if uploaded_file and api_key:
    with st.spinner("🔍 Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
    
    if pdf_text:
        with st.spinner("💬 Generating audio summary using Gemini..."):
            summary = generate_summary(pdf_text, api_key)

        st.subheader("📋 Summary")
        st.write(summary)

        with st.spinner("🎙️ Generating voiceover..."):
            audio_path = text_to_audio(summary)

        st.audio(audio_path, format="audio/mp3")
        
        # Download link for audio
        with open(audio_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:audio/mp3;base64,{b64}" download="summary.mp3">📥 Download Audio</a>'
            st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning("Couldn't extract text from PDF 😢 Try another file?")
elif uploaded_file and not api_key:
    st.warning("Please enter your Gemini API key in the sidebar!")

