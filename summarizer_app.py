import streamlit as st
from transformers import pipeline

# Load model once and cache it
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_model()

# Sample input abstracts (10 topics)
sample_texts = {
    "AI in Healthcare": """Artificial intelligence (AI) is transforming healthcare by enabling more accurate diagnoses, personalized treatment plans, and improved patient outcomes. AI systems are trained on large datasets and can detect patterns that may be difficult for humans to recognize. However, ethical considerations around data privacy, bias, and transparency remain significant challenges.""",
    
    "Quantum Computing": """Quantum computing leverages the principles of quantum mechanics to perform computations. Unlike classical bits, quantum bits or qubits can exist in multiple states simultaneously, offering exponential speed-ups for specific problems. Practical implementation, however, remains difficult due to decoherence and error rates.""",

    "Climate Change": """Climate change is causing increasingly severe weather events, rising sea levels, and disruptions to ecosystems. It poses a threat to global food security and water availability. Mitigating these effects requires global cooperation and policies focused on sustainable development and carbon emission reduction.""",

    "CRISPR Gene Editing": """CRISPR-Cas9 is a revolutionary gene-editing technology that allows for precise, targeted changes to the DNA of living organisms. This technology holds promise for treating genetic diseases, enhancing agriculture, and understanding gene functions, but it also raises ethical and safety concerns.""",

    "Blockchain Technology": """Blockchain is a distributed ledger technology that enables secure and transparent transactions without the need for intermediaries. Initially popularized by cryptocurrencies, it is now being explored in finance, supply chains, and healthcare for its potential to enhance security and reduce fraud.""",

    "5G Wireless Networks": """5G is the fifth generation of wireless communication technology, offering faster speeds, lower latency, and greater capacity than its predecessors. It is expected to enable innovations such as autonomous vehicles, smart cities, and advanced IoT applications.""",

    "Renewable Energy Adoption": """The transition to renewable energy sources such as solar, wind, and hydroelectric power is essential to combat climate change. While costs are falling and adoption is increasing, challenges remain in storage, grid integration, and political will.""",

    "Mental Health and Technology": """Digital technologies, including apps and wearables, are increasingly used to monitor and support mental health. While they offer accessibility and real-time data, concerns remain about data privacy, effectiveness, and reliance on non-human support.""",

    "Space Exploration": """Recent advancements in space exploration, including missions to Mars and private space travel, reflect a renewed global interest in space. While offering opportunities for scientific discovery and commercial growth, challenges include cost, safety, and long-term sustainability.""",

    "NLP and Language Models": """Natural Language Processing (NLP) has seen tremendous growth with models like GPT and BERT, which can generate and understand human language. These models are transforming industries, but issues like bias, hallucination, and control still persist."""
}

# 5 explanation styles
style_options = {
    "Simple": "Explain this in simple terms: ",
    "Technical": "Provide a technical summary: ",
    "Creative": "Summarize in a creative and engaging way: ",
    "Bullet Points": "Summarize this using bullet points: ",
    "For a 10-year-old": "Explain this so that a 10-year-old can understand: "
}

# Length settings
length_settings = {
    "Short": 150,
    "Medium": 400,
    "Long": 800
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
    max_len = length_settings[selected_length_key]

    # Prompt with style
    full_input = prefix + base_text

    with st.spinner("Generating summary..."):
        summary = summarizer(
            full_input,
            max_length=max_len,
            min_length=30,
            do_sample=False
        )[0]["summary_text"]

    st.subheader("📄 Summary")
    st.write(summary)
