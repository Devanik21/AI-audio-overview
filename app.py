import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
import base64
import google.generativeai as genai
from gtts import gTTS

# --- Page Setup ---
st.set_page_config(page_title=" PDF to Audio Summary", layout="centered",page_icon="🔊")
st.title("🎧 Gemini-Powered PDF Audio Overview")

# --- Sidebar for Gemini API Key ---
# --- Sidebar for Gemini API Key and Language ---
with st.sidebar:
    st.title("🔐 Gemini API")
    api_key = st.text_input("Enter your Gemini API key:", type="password")

    st.markdown("🌍 Select Voiceover Language")
    languages = {
        "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
        "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Chinese (Mandarin)": "zh-CN",
        "Japanese": "ja", "Korean": "ko", "Arabic": "ar", "Turkish": "tr", "Bengali": "bn",
        "Tamil": "ta", "Telugu": "te", "Gujarati": "gu", "Malayalam": "ml", "Urdu": "ur",
        "Indonesian": "id"
    }
    selected_lang = st.selectbox("Choose language for audio summary", list(languages.keys()))
    lang_code = languages[selected_lang]

    st.markdown("---")
    st.markdown("Made with 💖 by Prince")

# --- Updated TTS function ---
def text_to_audio(text, lang='en', filename="summary.mp3"):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    return filename


# --- Upload PDF ---
uploaded_file = st.file_uploader("📤 Upload a PDF", type=["pdf"])

if uploaded_file:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.success("✅ File Uploaded")
    with col2:
        st.markdown(f"**Filename:** `{uploaded_file.name}`")

    # Only generate when button is clicked
    if st.button("🎧 Generate Audio Overview"):
        if not api_key:
            st.warning("Please enter your Gemini API key in the sidebar!")
        else:
            with st.spinner("📖 Extracting PDF content..."):
                pdf_text, pages = extract_text_from_pdf(uploaded_file)

            if pdf_text:
                st.info(f"📄 Total Pages: {pages}")
                with st.spinner("🤖 Creating Gemini summary..."):
                    summary = generate_summary(pdf_text, api_key)

                st.subheader("📋 Summary")
                st.write(summary)

                with st.spinner("🎙️ Generating voiceover..."):
                    audio_path = text_to_audio(summary, lang=lang_code)


                st.success("✅ Audio overview ready!")
                st.audio(audio_path, format="audio/mp3")

                # Download Button
                with open(audio_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    st.markdown(
                        f'<a href="data:audio/mp3;base64,{b64}" download="summary.mp3">📥 Download Audio</a>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("😢 Couldn't extract text from the PDF. Try another file?")
