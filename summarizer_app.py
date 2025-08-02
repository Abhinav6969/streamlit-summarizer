import streamlit as st
from transformers import pipeline

# Load model once and cache it
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_model()

# Sample input abstracts (same as before, no changes)
sample_texts = {
    # [Your 10 paper entries here — unchanged for brevity]
    "AI in Healthcare": """Artificial intelligence (AI)...""",
    "Quantum Computing": """Quantum computing leverages...""",
    # Add the rest here...
}

# 5 explanation styles
style_options = {
    "Simple": "Explain this in simple terms: ",
    "Technical": "Provide a technical summary: ",
    "Creative": "Summarize in a creative and engaging way: ",
    "Bullet Points": "Summarize this using bullet points: ",
    "For a 10-year-old": "Explain this so that a 10-year-old can understand: "
}

# Length settings: define both min and max
length_settings = {
    "Short": {"min": 200, "max": 300},
    "Medium": {"min": 500, "max": 800},
    "Long": {"min": 1000, "max": 2000}
}

# Streamlit UI
st.title(" Hugging Face Research Paper Summarizer")

selected_text_key = st.selectbox(" Select a research topic:", list(sample_texts.keys()))
selected_style_key = st.selectbox(" Choose explanation style:", list(style_options.keys()))
selected_length_key = st.selectbox(" Select summary length:", list(length_settings.keys()))

# On button click
if st.button("Summarize"):
    base_text = sample_texts[selected_text_key]
    prefix = style_options[selected_style_key]

    min_len = length_settings[selected_length_key]["min"]
    max_len = length_settings[selected_length_key]["max"]

    full_input = prefix + base_text

    with st.spinner("Generating summary..."):
        summary = summarizer(
            full_input,
            min_length=min_len,
            max_length=max_len,
            do_sample=False
        )[0]["summary_text"]

    st.subheader("📄 Summary")
    st.write(summary)
