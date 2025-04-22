import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
import base64
import google.generativeai as genai
from gtts import gTTS
from langdetect import detect

# --- Page Setup ---
st.set_page_config(page_title="\ud83d\udcc4\ud83d\udd0a PDF to Audio Summary", layout="centered")
st.title("\ud83d\udcac\ud83c\udfbf Gemini-Powered PDF Audio Overview")

st.markdown("""
<style>
    .step-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar for Gemini API Key and Language ---
with st.sidebar:
    st.title("\ud83d\udd10 Gemini API")
    api_key = st.text_input("Enter your Gemini API key:", type="password")

    st.markdown("\u2728 Language Options")
    lang_auto = st.checkbox("Auto-detect language from PDF", value=True)

    languages = {
        "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
        "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Chinese (Mandarin)": "zh-CN",
        "Japanese": "ja", "Korean": "ko", "Arabic": "ar", "Turkish": "tr", "Bengali": "bn",
        "Tamil": "ta", "Telugu": "te", "Gujarati": "gu", "Malayalam": "ml", "Urdu": "ur",
        "Indonesian": "id"
    }

    selected_lang = st.selectbox("Choose language (if not auto-detecting):", list(languages.keys()), index=0)
    lang_code = languages[selected_lang]

    st.markdown("---")
    st.markdown("Made with \ud83d\udc96 by Prince")

# --- Function to extract text and page info ---
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    text = ""
    page_count = 0
    with fitz.open(tmp_file_path) as doc:
        page_count = len(doc)
        for page in doc:
            text += page.get_text()

    os.remove(tmp_file_path)
    return text.strip(), page_count

# --- Gemini Summary Function ---
def generate_summary(text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = "Summarize this PDF content in a friendly, voiceover-style overview:\n\n" + text
    response = model.generate_content(prompt)
    return response.text

# --- TTS Function ---
def text_to_audio(text, lang='en', filename="summary.mp3"):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)
    return filename

# --- Upload PDF ---
uploaded_file = st.file_uploader("\ud83d\udcc4 Upload a PDF", type=["pdf"])

if uploaded_file:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.success("\u2705 File Uploaded")
    with col2:
        st.markdown(f"**Filename:** `{uploaded_file.name}`")

    if st.button("\ud83c\udfbf Generate Audio Overview"):
        if not api_key:
            st.warning("Please enter your Gemini API key in the sidebar!")
        else:
            with st.spinner("\ud83d\udcd6 Extracting PDF content..."):
                pdf_text, pages = extract_text_from_pdf(uploaded_file)

            if pdf_text:
                word_count = len(pdf_text.split())
                st.markdown(f"<div class='step-box'>\ud83d\udcc4 Extracted <strong>{word_count}</strong> words from <strong>{pages}</strong> pages</div>", unsafe_allow_html=True)

                # --- Auto-detect language if enabled ---
                detected_lang = lang_code
                if lang_auto:
                    try:
                        detected_lang = detect(pdf_text)
                        st.markdown(f"<div class='step-box'>\ud83c\udf0d Detected Language: <strong>{detected_lang.upper()}</strong></div>", unsafe_allow_html=True)
                    except:
                        st.warning("\u26a0\ufe0f Language detection failed. Using selected language instead.")

                with st.spinner("\ud83e\udde0 Creating Gemini summary..."):
                    summary = generate_summary(pdf_text, api_key)
                st.markdown("<div class='step-box'>\ud83e\udd16 AI Summary Created</div>", unsafe_allow_html=True)

                st.subheader("\ud83d\udccb Summary")
                st.write(summary)

                with st.spinner("\ud83c\udf99\ufe0f Converting to speech..."):
                    audio_path = text_to_audio(summary, lang=detected_lang)
                st.markdown("<div class='step-box'>\ud83c\udfa7 Audio Conversion Done!</div>", unsafe_allow_html=True)

                st.success("\u2705 Processing Complete!")
                st.audio(audio_path, format="audio/mp3")

                # Download Button
                with open(audio_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    st.markdown(
                        f'<a href="data:audio/mp3;base64,{b64}" download="summary.mp3">\ud83d\udcc5 Download Audio</a>',
                        unsafe_allow_html=True
                    )
            else:
                st.warning("\ud83d\ude25 Couldn't extract text from the PDF. Try another file?")
