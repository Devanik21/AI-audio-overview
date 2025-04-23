import streamlit as st
import fitz  # PyMuPDF
import tempfile
import os
import base64
import google.generativeai as genai
from gtts import gTTS
from langdetect import detect, LangDetectException
from googletrans import Translator
import datetime
import json
import re
import uuid
import pandas as pd
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import tabula
import difflib
import hashlib

# ====== PAGE CONFIGURATION ======
st.set_page_config(page_title="PDF Audio Summarizer Pro", layout="wide")

# ====== STYLE ======
st.markdown("""
<style>
    .step-box { background: #0f2d42; padding: 1rem; border-radius: 12px;
                margin: 10px 0; border-left: 6px solid #3b82f6;
                font-size: 1.1rem; font-weight: 500; color: #0c4a6e; }
    .header-text { font-weight: 700; font-size: 2rem; color: #1e293b; }
    .subheader-text { font-weight: 600; color: #475569; margin-bottom: 1.5rem; }
    .footer-text { color: #94a3b8; font-size: 0.9rem; margin-top: 3rem; text-align: center; }
    .insight-box { background: #ecfdf5; padding: 0.8rem; border-radius: 8px; 
                  margin: 5px 0; border-left: 4px solid #10b981; }
    .citation-box { background: #f9fafb; padding: 1rem; border-radius: 8px;
                  border: 1px solid #e5e7eb; font-family: monospace; }
    .tab-content { padding: 20px 0; }
    .compare-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .compare-box { background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; }
    .toc-item { cursor: pointer; padding: 5px 0; transition: all 0.2s; }
    .toc-item:hover { background: #f1f5f9; border-radius: 4px; padding-left: 5px; }
    .history-card { background: white; padding: 15px; border-radius: 10px; 
                   box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# ====== SESSION STATE INITIALIZATION ======
if 'summary_history' not in st.session_state:
    st.session_state.summary_history = []

if 'current_pdf_hash' not in st.session_state:
    st.session_state.current_pdf_hash = None

# ====== SIDEBAR ======
with st.sidebar:
    st.header("Gemini API Settings")
    api_key = st.text_input("Enter your Gemini API key:", type="password")

    st.markdown("---")
    
    # === FEATURE 1: Multiple Language Summarization ===
    st.header("Language Options")
    auto_lang = st.checkbox("Auto-detect language from PDF", value=True)

    LANGUAGES = {
        "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
        "Italian": "it", "Portuguese": "pt", "Russian": "ru", "Chinese (Mandarin)": "zh-CN",
        "Japanese": "ja", "Korean": "ko", "Arabic": "ar", "Turkish": "tr", "Bengali": "bn",
        "Tamil": "ta", "Telugu": "te", "Gujarati": "gu", "Malayalam": "ml", "Urdu": "ur",
        "Indonesian": "id", "Vietnamese": "vi", "Polish": "pl", "Dutch": "nl", "Swedish": "sv"
    }

    chosen_lang = st.selectbox("Source language (if not auto-detecting):", list(LANGUAGES.keys()))
    lang_code = LANGUAGES[chosen_lang]
    
    translate_summary = st.checkbox("Translate summary to different language", value=False)
    if translate_summary:
        target_lang = st.selectbox("Target language for summary:", list(LANGUAGES.keys()))
        target_lang_code = LANGUAGES[target_lang]
    else:
        target_lang_code = None

    st.markdown("---")
    
    # === FEATURE 2: Voice Selection Options ===
    st.header("Voice Options")
    voice_gender = st.radio("Voice type:", ["Female", "Male"], horizontal=True)
    voice_style = st.select_slider("Speaking style:", 
                                  options=["Formal", "Standard", "Casual", "Enthusiastic"])
    speech_speed = st.slider("Speaking speed:", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    
    # === FEATURE 6: Customizable Summary Length ===
    st.markdown("---")
    st.header("Summary Options")
    summary_length = st.select_slider("Summary length:", 
                                     options=["Very Brief", "Brief", "Standard", "Detailed", "Comprehensive"])

    st.markdown("---")
    audio_format = st.selectbox("Choose audio format:", ["mp3", "ogg", "wav"])
    download_transcript = st.checkbox("Download Summary as Text File")
    show_wordcloud = st.checkbox("Show WordCloud of PDF Text")
    
    # === FEATURE 3: Key Insights Extraction ===
    extract_key_insights = st.checkbox("Extract key insights", value=True)
    
    # === FEATURE 5: Visual Data Extraction ===
    extract_visual_data = st.checkbox("Describe tables and charts", value=True)
    
    # === FEATURE 9: Citation and Reference Extraction ===
    extract_citations = st.checkbox("Extract citations and references", value=True)
    
    st.markdown("<small>Updated on April 2025</small>", unsafe_allow_html=True)

# ====== FUNCTIONS ======
def generate_pdf_hash(uploaded_pdf):
    """Generate unique hash for PDF file to identify it"""
    content = uploaded_pdf.getvalue()
    return hashlib.md5(content).hexdigest()

def extract_text_from_pdf(uploaded_pdf):
    """Extract text content from PDF"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp_path = tmp.name
    text = ""
    toc = []
    headings = []
    images_count = 0
    tables_count = 0
    
    with fitz.open(tmp_path) as doc:
        # Try to extract the table of contents
        try:
            toc = doc.get_toc()
        except:
            pass
            
        for page_num, page in enumerate(doc):
            # Extract text
            text += page.get_text()
            
            # Count images
            images = page.get_images(full=True)
            images_count += len(images)
            
            # Extract headings heuristically
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                if span.get("size", 0) > 12 and span.get("text", "").strip():
                                    # Likely a heading
                                    headings.append({
                                        "text": span["text"].strip(),
                                        "page": page_num + 1
                                    })
        
        page_count = len(doc)
    
    # Try to extract tables
    try:
        tables = tabula.read_pdf(tmp_path, pages='all')
        tables_count = len(tables)
    except:
        tables_count = 0
    
    os.remove(tmp_path)
    
    # If no TOC extracted, use headings
    if not toc and headings:
        toc = [(1, h["text"], h["page"]) for h in headings]
    
    return text.strip(), page_count, toc, images_count, tables_count

def chunk_text(text, max_tokens=1500):
    """Split text into manageable chunks"""
    words = text.split()
    return [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def translate_text(text, target_language):
    """Translate text to target language"""
    translator = Translator()
    result = translator.translate(text, dest=target_language)
    return result.text

def extract_sections_by_toc(text, toc):
    """Extract sections of text based on table of contents"""
    sections = {}
    if not toc:
        sections["Full Document"] = text
        return sections
    
    # Sort TOC by page number
    sorted_toc = sorted(toc, key=lambda x: x[2])
    
    for i, (level, title, page) in enumerate(sorted_toc):
        # For simplicity, we'll use regex to find sections
        section_start = re.search(re.escape(title), text)
        if section_start:
            start_idx = section_start.start()
            
            # Find the end of this section (start of next section)
            end_idx = len(text)
            if i < len(sorted_toc) - 1:
                next_title = sorted_toc[i+1][1]
                next_section = re.search(re.escape(next_title), text[start_idx+len(title):])
                if next_section:
                    end_idx = start_idx + len(title) + next_section.start()
            
            sections[title] = text[start_idx:end_idx]
    
    # If no sections were found
    if not sections:
        sections["Full Document"] = text
        
    return sections

def extract_figures_and_tables(pdf_path):
    """Extract descriptions of figures and tables"""
    descriptions = []
    
    # Extract tables using tabula
    try:
        tables = tabula.read_pdf(pdf_path, pages='all')
        for i, table in enumerate(tables):
            if not table.empty:
                desc = f"Table {i+1} contains {len(table)} rows and {len(table.columns)} columns. "
                desc += f"Columns include: {', '.join(table.columns.astype(str))}. "
                descriptions.append(desc)
    except:
        pass
    
    # Use PyMuPDF to identify figures
    doc = fitz.open(pdf_path)
    for page_idx, page in enumerate(doc):
        # Get images
        images = page.get_images(full=True)
        for img_idx, img in enumerate(images):
            descriptions.append(f"Figure found on page {page_idx+1}, likely contains a chart, diagram, or image.")
    
    return descriptions

def extract_citations(text):
    """Extract possible citations and references from text"""
    citations = []
    
    # Look for common citation patterns
    # APA style (Author, Year)
    apa_matches = re.findall(r'\(([A-Za-z]+(?:[ ,&]+[A-Za-z]+)*),? (\d{4}[a-z]?)\)', text)
    # MLA style "Author"
    mla_matches = re.findall(r'"([^"]+)" \(([A-Za-z]+(?:[ ,&]+[A-Za-z]+)*)\)', text)
    # URL citations
    url_matches = re.findall(r'https?://[^\s)<>"]+', text)
    # DOI
    doi_matches = re.findall(r'doi:[\w\./]+', text)
    # Chicago Style
    chicago_matches = re.findall(r'\d+\. [A-Z][a-z]+, [A-Z][a-z]+', text)
    
    # References section
    ref_section = None
    ref_headers = ["References", "Bibliography", "Works Cited", "Literature", "Sources"]
    for header in ref_headers:
        ref_match = re.search(f"{header}.*?\n(.+?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
        if ref_match:
            ref_section = ref_match.group(1)
            break
    
    # Combine all findings
    if apa_matches:
        citations.extend([f"{author} ({year})" for author, year in apa_matches[:20]])
    if mla_matches:
        citations.extend([f'"{quote}" ({author})' for quote, author in mla_matches[:20]])
    if url_matches:
        citations.extend(url_matches[:20])
    if doi_matches:
        citations.extend(doi_matches[:20])
    if chicago_matches:
        citations.extend(chicago_matches[:20])
    
    # If ref section found, add it
    if ref_section:
        # Split by line breaks and add each reference
        refs = [r.strip() for r in ref_section.split('\n') if r.strip()]
        citations.extend(refs[:30])  # Limit to 30 references
    
    # Remove duplicates
    citations = list(set(citations))
    
    return citations

def generate_ai_summary(text, key, length='Standard', extract_insights=False, visual_descriptions=None):
    """Generate AI summary with appropriate prompts based on settings"""
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    length_prompt = {
        "Very Brief": "Create an extremely concise summary in just 2-3 sentences.",
        "Brief": "Create a brief summary focusing only on main points.",
        "Standard": "Create a balanced summary of moderate length.",
        "Detailed": "Create a detailed summary covering most important information.",
        "Comprehensive": "Create a comprehensive summary covering all significant details."
    }
    
    chunks = chunk_text(text)
    summaries = []
    
    # Create base prompt
    base_prompt = f"{length_prompt[length]}\nSummarize this in a friendly voiceover style:\n"
    
    # Process each chunk
    for c in chunks:
        summaries.append(model.generate_content(f"{base_prompt}{c}").text)
    
    full_summary = "\n\n".join(summaries)
    
    results = {"summary": full_summary}
    
    # Extract key insights if requested
    if extract_insights:
        insights_prompt = "Extract the 5-7 most important insights or key points from this text as a bulletpoint list:"
        insights = model.generate_content(f"{insights_prompt}\n{text[:5000]}").text
        results["insights"] = insights
    
    # Process visual descriptions if provided
    if visual_descriptions and len(visual_descriptions) > 0:
        visual_prompt = "Based on these descriptions of visual elements in the document, provide a brief explanation of what they likely show:"
        visual_text = "\n".join(visual_descriptions)
        visual_summary = model.generate_content(f"{visual_prompt}\n{visual_text}").text
        results["visual_summary"] = visual_summary
    
    return results

def answer_question(text, question, key):
    """Generate answer to specific question about the document"""
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    prompt = f"""Based on the following document, please answer this question as accurately as possible: 
    Question: {question}
    
    Document text:
    {text[:10000]}  # Using first portion to stay within token limits
    """
    
    try:
        response = model.generate_content(prompt).text
        return response
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def compare_pdfs(text1, text2):
    """Compare two PDF texts and identify similarities and differences"""
    # Split into lines for better comparison
    lines1 = text1.split('\n')
    lines2 = text2.split('\n')
    
    # Get diff
    d = difflib.Differ()
    diff = list(d.compare(lines1, lines2))
    
    # Process differences
    similar = []
    only_in_first = []
    only_in_second = []
    
    for line in diff:
        if line.startswith('  '):  # Common line
            similar.append(line[2:])
        elif line.startswith('- '):  # Only in first file
            only_in_first.append(line[2:])
        elif line.startswith('+ '):  # Only in second file
            only_in_second.append(line[2:])
    
    # Calculate similarity percentage
    similarity = len(similar) / max(len(lines1), len(lines2)) * 100
    
    return {
        "similarity_percentage": round(similarity, 2),
        "similarities": '\n'.join(similar[:50]),  # First 50 similarities
        "only_in_first": '\n'.join(only_in_first[:50]),  # First 50 differences
        "only_in_second": '\n'.join(only_in_second[:50])  # First 50 differences
    }

def text_to_speech(text, language, filename, speed=1.0, gender="Female", style="Standard"):
    """Convert text to speech with voice customization"""
    # Note: Basic gTTS doesn't support all these parameters, but we're including them
    # for future expansion with more advanced TTS systems
    
    # Apply voice style modifications to text (simple implementation)
    if style == "Formal":
        text = text.replace("hey", "greetings").replace("yeah", "yes").replace("okay", "indeed")
    elif style == "Casual":
        text = text.replace("additionally", "also").replace("furthermore", "plus")
    elif style == "Enthusiastic":
        text = text + " " + "!".join(text.split(".")[:3])  # Add some excitement to first sentences
    
    # Create TTS with available parameters
    tts = gTTS(text=text, lang=language, slow=(speed < 0.9))
    tts.save(filename)
    return filename

def create_audio_download_link(audio_file, format_type):
    """Create download link for audio file"""
    with open(audio_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:audio/{format_type};base64,{b64}" download="summary.{format_type}" class="download-button">Download Audio</a>'

def estimate_audio_duration(word_count, speed=1.0):
    """Estimate audio duration based on word count and speed"""
    wpm = 130 * speed  # Adjust words per minute based on speed
    minutes = word_count / wpm
    return f"{int(minutes)} min {int((minutes % 1) * 60)} sec"

def save_transcript(summary):
    """Save and create download link for transcript"""
    filename = f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(summary)
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    os.remove(filename)
    return f'<a href="data:text/plain;base64,{b64}" download="summary.txt" class="download-button">Download Transcript</a>'

def show_wordcloud_img(text):
    """Generate and display word cloud from text"""
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Display word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    st.pyplot(fig)

def save_to_history(pdf_name, summary, insights=None, total_pages=0):
    """Save summary to history"""
    history_item = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pdf_name": pdf_name,
        "summary": summary,
        "insights": insights,
        "pages": total_pages
    }
    
    st.session_state.summary_history.append(history_item)
    
    # Limit history to most recent 10 items
    if len(st.session_state.summary_history) > 10:
        st.session_state.summary_history = st.session_state.summary_history[-10:]

# ====== MAIN UI ======
st.markdown('<p class="header-text">PDF to Audio Summary Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Upload PDFs to get intelligent spoken summaries powered by AI.</p>', unsafe_allow_html=True)

# Create tabs for main functionality
tab1, tab2, tab3, tab4 = st.tabs(["📄 Summarize PDF", "❓ Q&A Mode", "🔍 Compare PDFs", "📚 History"])

# Tab 1: Main PDF Summarization
with tab1:
    uploaded_pdf = st.file_uploader("Upload your PDF file", type=["pdf"], key="main_pdf")

    if uploaded_pdf:
        # Generate hash for current PDF
        pdf_hash = generate_pdf_hash(uploaded_pdf)
        st.session_state.current_pdf_hash = pdf_hash
        
        st.info(f"📄 File selected: **{uploaded_pdf.name}**")
        if st.button("Generate Audio Overview"):
            if not api_key:
                st.error("⚠️ Please enter your Gemini API key in the sidebar.")
            else:
                with st.spinner("Extracting text from PDF..."):
                    pdf_text, total_pages, toc, images_count, tables_count = extract_text_from_pdf(uploaded_pdf)
                
                word_count = len(pdf_text.split())
                st.markdown(f'<div class="step-box">Extracted <b>{word_count}</b> words from <b>{total_pages}</b> pages. Found {images_count} images and {tables_count} tables.</div>', unsafe_allow_html=True)

                if show_wordcloud:
                    with st.spinner("Generating WordCloud..."):
                        show_wordcloud_img(pdf_text)

                # === FEATURE 4: Interactive Table of Contents ===
                if toc:
                    st.markdown("### Table of Contents")
                    st.info("Click on a section to get a specific summary for that part.")
                    
                    sections = extract_sections_by_toc(pdf_text, toc)
                    selected_section = st.selectbox("Jump to section:", 
                                                  ["Full Document"] + [title for _, title, _ in toc])
                    
                    if selected_section != "Full Document":
                        section_text = sections.get(selected_section, "")
                        if section_text:
                            pdf_text = section_text
                            st.success(f"Now summarizing section: {selected_section}")

                if auto_lang:
                    try:
                        detected_lang = detect(pdf_text)
                        st.markdown(f'<div class="step-box">Detected language: <b>{detected_lang.upper()}</b></div>', unsafe_allow_html=True)
                    except LangDetectException:
                        detected_lang = lang_code
                        st.warning("Failed to detect language, using selected option.")
                else:
                    detected_lang = lang_code

                # === FEATURE 5: Visual Data Extraction ===
                visual_descriptions = None
                if extract_visual_data and (images_count > 0 or tables_count > 0):
                    with st.spinner("Analyzing visual elements..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_pdf.getvalue())
                            tmp_path = tmp.name
                            visual_descriptions = extract_figures_and_tables(tmp_path)
                        os.remove(tmp_path)

                with st.spinner("Generating AI summary..."):
                    try:
                        ai_output = generate_ai_summary(
                            pdf_text, 
                            api_key, 
                            length=summary_length,
                            extract_insights=extract_key_insights,
                            visual_descriptions=visual_descriptions
                        )
                        
                        summary = ai_output["summary"]
                        
                        # Translate if requested
                        if translate_summary and target_lang_code:
                            with st.spinner(f"Translating to {target_lang}..."):
                                summary = translate_text(summary, target_lang_code)
                                if "insights" in ai_output:
                                    ai_output["insights"] = translate_text(ai_output["insights"], target_lang_code)
                    except Exception as e:
                        st.error(f"Summary generation failed: {e}")
                        st.stop()

                st.markdown('<div class="step-box">AI Summary generated successfully.</div>', unsafe_allow_html=True)
                
                # === FEATURE 3: Key Insights Display ===
                if extract_key_insights and "insights" in ai_output:
                    st.subheader("Key Insights")
                    insights = ai_output["insights"]
                    # Format insights into nice boxes
                    for line in insights.split("\n"):
                        if line.strip():
                            # Remove leading bullet points
                            clean_line = re.sub(r'^[\s\-\*•]+', '', line).strip()
                            if clean_line:
                                st.markdown(f'<div class="insight-box">{clean_line}</div>', unsafe_allow_html=True)
                
                # Show visual data summary if available
                if extract_visual_data and "visual_summary" in ai_output:
                    st.subheader("Visual Elements")
                    st.write(ai_output["visual_summary"])
                
                st.subheader("Full Summary")
                st.write(summary)
                duration = estimate_audio_duration(len(summary.split()), speech_speed)
                st.caption(f"Estimated audio length: {duration}")

                if download_transcript:
                    transcript_link = save_transcript(summary)
                    st.markdown(transcript_link, unsafe_allow_html=True)

                # === FEATURE 9: Citation and Reference Extraction ===
                if extract_citations:
                    citations = extract_citations(pdf_text)
                    if citations:
                        st.subheader("Citations & References")
                        with st.expander("View all citations", expanded=False):
                            st.markdown('<div class="citation-box">', unsafe_allow_html=True)
                            for citation in citations:
                                st.write(f"• {citation}")
                            st.markdown('</div>', unsafe_allow_html=True)

                with st.spinner("Converting to speech..."):
                    audio_filename = f"summary.{audio_format}"
                    final_lang = target_lang_code if translate_summary else detected_lang
                    audio_file = text_to_speech(
                        summary, 
                        final_lang, 
                        audio_filename,
                        speed=speech_speed,
                        gender=voice_gender,
                        style=voice_style
                    )

                st.markdown('<div class="step-box">Audio conversion completed!</div>', unsafe_allow_html=True)
                st.success("✅ Done!")

                st.audio(audio_file, format=f"audio/{audio_format}")
                download_link = create_audio_download_link(audio_file, audio_format)
                st.markdown(download_link, unsafe_allow_html=True)

                # === FEATURE 10: Summary History ===
                # Save to history
                save_to_history(
                    uploaded_pdf.name, 
                    summary, 
                    ai_output.get("insights"), 
                    total_pages
                )

                try:
                    os.remove(audio_file)
                except Exception:
                    pass

# === FEATURE 7: Question & Answer Mode ===
with tab2:
    st.subheader("Ask Questions About Your PDF")
    st.write("Upload a PDF and ask specific questions about its content.")
    
    uploaded_pdf_qa = st.file_uploader("Upload your PDF file", type=["pdf"], key="qa_pdf")
    
    if uploaded_pdf_qa:
        st.info(f"📄 File selected: **{uploaded_pdf_qa.name}**")
        
        # Extract text when PDF is uploaded
        if 'qa_text' not in st.session_state:
            with st.spinner("Extracting text from PDF..."):
                qa_text, qa_pages, _, _, _ = extract_text_from_pdf(uploaded_pdf_qa)
                st.session_state.qa_text = qa_text
                st.markdown(f'<div class="step-box">Extracted content from <b>{qa_pages}</b> pages.</div>', unsafe_allow_html=True)
        
        # Question input
        user_question = st.text_input("Ask a question about the document:")
        
        if user_question and api_key:
            if st.button("Get Answer"):
                with st.spinner("Finding answer..."):
                    answer = answer_question(st.session_state.qa_text, user_question, api_key)
                    st.markdown(f"### Answer\n{answer}")
        elif user_question:
            st.warning("Please enter your Gemini API key in the sidebar.")

# === FEATURE 8: Multi-PDF Comparison ===
with tab3:
    st.subheader("Compare Two PDFs")
    st.write("Upload two PDFs to find similarities and differences.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_pdf1 = st.file_uploader("Upload first PDF", type=["pdf"], key="compare_pdf1")
    
    with col2:
        uploaded_pdf2 = st.file_uploader("Upload second PDF", type=["pdf"], key="compare_pdf2")
    
    if uploaded_pdf1 and uploaded_pdf2:
        if st.button("Compare Documents"):
            with st.spinner("Analyzing documents..."):
                # Extract text from both PDFs
                text1, pages1, _, _, _ = extract_text_from_pdf(uploaded_pdf1)
                text2, pages2, _, _, _ = extract_text_from_pdf(uploaded_pdf2)
                
                # Compare texts
                comparison = compare_pdfs(text1, text2)
                
                # Display results
                st.success(f"Analysis complete! Documents are {comparison['similarity_percentage']}% similar.")
                
                st.markdown("### Comparison Results")
                
                # Similarity percentage with gauge chart
                similarity = comparison['similarity_percentage']
                similarity_color = "green" if similarity > 70 else "orange" if similarity > 40 else "red"
                
                # Create simple gauge chart
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh(0, similarity, color=similarity_color)
                ax.barh(0, 100, color="lightgrey", alpha=0.3)
                ax.set_xlim(0, 100)
                ax.set_yticks([])
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_title(f"Similarity: {similarity}%")
                st.pyplot(fig)
                
                # Display content comparisons
                st.markdown('<div class="compare-container">', unsafe_allow_html=True)
                
                # Column for first document
                st.markdown('<div class="compare-box">', unsafe_allow_html=True)
                st.markdown(f"#### Unique to {uploaded_pdf1.name}")
                st.write(comparison['only_in_first'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Column for second document
                st.markdown('<div class="compare-box">', unsafe_allow_html=True)
                st.markdown(f"#### Unique to {uploaded_pdf2.name}")
                st.write(comparison['only_in_second'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Common content
                st.markdown("#### Common Content")
                st.write(comparison['similarities'])
