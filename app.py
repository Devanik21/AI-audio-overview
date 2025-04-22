import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
import base64
import google.generativeai as genai
from gtts import gTTS

# Page configuration with custom theme
st.set_page_config(
    page_title="PDF to Audio Summary",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stButton>button {
        background-color: #5e72e4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4454c3;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #344767;
    }
    .upload-section {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
    .results-section {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .sidebar .css-1d391kg {
        background-color: #f1f3f9;
    }
    .stProgress .st-bo {
        background-color: #5e72e4;
    }
    .api-input {
        margin-top: 1rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    text = ""
    with fitz.open(tmp_file_path) as doc:
        page_count = len(doc)
        for page in doc:
            text += page.get_text()
    
    os.remove(tmp_file_path)
    return text.strip(), page_count

def generate_summary(text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    Create a clear, engaging, and professional voiceover-style summary of this PDF content.
    Focus on key points and maintain a friendly, conversational tone.
    Keep the summary concise but comprehensive.
    
    PDF CONTENT:
    {text}
    """
    response = model.generate_content(prompt)
    return response.text

def text_to_audio(text, filename="summary.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

def create_download_link(audio_path, filename="summary.mp3"):
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    download_link = f'''
    <a href="data:audio/mp3;base64,{b64}" download="{filename}" 
       style="text-decoration:none;">
        <div style="background-color:#5e72e4; color:white; padding:12px 20px; 
                    border-radius:8px; text-align:center; margin-top:15px; 
                    display:flex; align-items:center; justify-content:center; 
                    font-weight:bold; cursor:pointer; transition: all 0.3s">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" 
                 style="margin-right:8px" viewBox="0 0 16 16">
                <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
            </svg>
            Download Audio File
        </div>
    </a>
    '''
    return download_link

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150x80?text=AudioPDF", width=150)
    st.title("🎧 AudioPDF")
    
    st.markdown("#### Settings")
    with st.container():
        st.markdown('<div class="api-input">', unsafe_allow_html=True)
        api_key = st.text_input("🔑 Gemini API Key", type="password", 
                               help="Enter your Gemini API key here")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    ### How it works
    1. Upload your PDF
    2. Enter your Gemini API key
    3. Generate an audio summary
    4. Listen or download
    """)
    
    st.markdown("---")
    st.caption("© 2025 AudioPDF | v1.2.0")

# Main content
st.markdown('<h1 style="text-align:center; margin-bottom:1.5rem">🎧 PDF to Audio Summary</h1>', 
           unsafe_allow_html=True)

# Upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
col1, col2 = st.columns([3, 2])
with col1:
    st.markdown("### Upload your PDF document")
    st.markdown("We'll transform it into an engaging audio summary using Gemini AI.")
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

with col2:
    st.image("https://via.placeholder.com/300x200?text=PDF+to+Audio", width=300)
st.markdown('</div>', unsafe_allow_html=True)

# Processing section
if uploaded_file:
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    
    # File info display
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("### File Details")
    with col2:
        st.markdown(f"**Filename:** `{uploaded_file.name}`")
        file_size = round(uploaded_file.size / 1024, 1)
        st.markdown(f"**Size:** `{file_size} KB`")
    with col3:
        if st.button("🎧 Generate Audio", use_container_width=True):
            if not api_key:
                st.error("⚠️ Please enter your Gemini API key in the sidebar!")
            else:
                # Extract text with progress bar
                with st.status("Processing your PDF...", expanded=True) as status:
                    st.write("📄 Extracting text...")
                    pdf_text, pages = extract_text_from_pdf(uploaded_file)
                    
                    if not pdf_text:
                        st.error("❌ Could not extract text from this PDF. Please try another file.")
                    else:
                        st.write(f"✅ Extracted {len(pdf_text.split())} words from {pages} pages")
                        
                        # Generate summary
                        st.write("🤖 Creating AI summary...")
                        summary = generate_summary(pdf_text, api_key)
                        
                        # Create audio
                        st.write("🎙️ Converting to speech...")
                        audio_filename = f"{uploaded_file.name.split('.')[0]}_summary.mp3"
                        audio_path = text_to_audio(summary, audio_filename)
                        
                        status.update(label="✅ Processing complete!", state="complete")
                
                        # Display results
                        st.markdown("### 📝 Summary")
                        st.write(summary)
                        
                        st.markdown("### 🔊 Audio Overview")
                        st.audio(audio_path, format="audio/mp3")
                        
                        # Download button
                        st.markdown(create_download_link(audio_path, audio_filename), 
                                   unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Instructions when no file is uploaded
    st.info("👆 Please upload a PDF document to get started")
