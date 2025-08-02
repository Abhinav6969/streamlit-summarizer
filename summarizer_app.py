import streamlit as st
from transformers import pipeline

# Load model once and cache it
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

summarizer = load_model()


sample_texts = {
    "AI in Healthcare": """Artificial Intelligence (AI) is revolutionizing the healthcare industry by enhancing the accuracy of diagnostics, improving treatment personalization, and streamlining hospital workflows. For example, machine learning algorithms can analyze radiological images to detect tumors at earlier stages than human radiologists. AI is also being used to predict patient deterioration, manage chronic diseases, and optimize hospital resource allocation. Despite these advancements, ethical issues remain prominent, particularly in relation to data privacy, algorithmic bias, and lack of transparency. Ensuring that AI systems do not perpetuate existing healthcare disparities is critical. Additionally, regulatory frameworks must evolve to accommodate the rapid development of AI technologies and ensure they are used safely and effectively in clinical settings.""",

    "Quantum Computing": """Quantum computing represents a paradigm shift in computational power by leveraging the principles of quantum mechanics. Qubits, which can exist in superposition and entanglement, allow quantum computers to process vast amounts of information simultaneously. This technology promises breakthroughs in cryptography, optimization, and material science. For instance, quantum algorithms like Shor’s algorithm could break classical encryption methods, posing both opportunities and risks for cybersecurity. However, the field still faces major challenges such as quantum decoherence, error correction, and scalability of quantum hardware. Developing stable, fault-tolerant quantum systems remains a key hurdle. Governments and private companies alike are investing billions into quantum research, recognizing its potential to disrupt current computational paradigms.""",

    "Climate Change": """Climate change, driven primarily by the accumulation of greenhouse gases like carbon dioxide and methane in the atmosphere, is causing a multitude of environmental disruptions. These include more frequent and severe natural disasters such as hurricanes and wildfires, rising global temperatures, melting polar ice caps, and ocean acidification. The consequences extend to agriculture, water resources, and human health, threatening food and water security and increasing the prevalence of diseases. Efforts to combat climate change involve mitigation strategies like transitioning to renewable energy, improving energy efficiency, and reforestation, as well as adaptation strategies such as building resilient infrastructure. International cooperation, particularly through agreements like the Paris Accord, is essential to reduce emissions and limit global warming to below 1.5°C.""",

    "CRISPR Gene Editing": """CRISPR-Cas9 is a revolutionary tool for genome editing that enables scientists to alter DNA sequences with unprecedented precision and efficiency. Originally derived from a bacterial defense mechanism, CRISPR allows for targeted modification of genes, offering potential cures for genetic disorders such as cystic fibrosis and sickle cell anemia. It is also being used in agriculture to create crops that are more resistant to pests and environmental stress. However, the ethical implications of gene editing, especially germline editing that affects future generations, have sparked significant debate. Concerns include the potential for unintended consequences, genetic inequality, and 'designer babies.' Regulatory bodies worldwide are grappling with how to oversee the safe and ethical use of CRISPR technology.""",

    "Blockchain Technology": """Blockchain is a decentralized digital ledger technology that enables secure, transparent, and tamper-proof record-keeping without relying on a central authority. Each block in the chain contains a list of transactions, and once added, blocks cannot be altered retroactively without changing all subsequent blocks. Initially used for cryptocurrencies like Bitcoin, blockchain is now being applied across sectors including finance, supply chain management, and healthcare. It facilitates greater transparency and reduces fraud in complex, multi-party transactions. Despite its potential, blockchain adoption faces challenges such as high energy consumption (especially in proof-of-work systems), scalability issues, and regulatory uncertainties. The development of more efficient consensus mechanisms like proof-of-stake may help address some of these concerns.""",

    "5G Wireless Networks": """5G technology represents the next generation of mobile communications, offering significantly faster data speeds, lower latency, and the ability to connect a massive number of devices simultaneously. These improvements are expected to accelerate innovations such as autonomous vehicles, smart cities, augmented reality, and remote surgery. Unlike previous generations, 5G relies on a dense infrastructure of small cell towers and operates across multiple frequency bands. While it holds immense promise, the rollout of 5G faces logistical and regulatory challenges, including spectrum allocation and public concerns about health effects. Moreover, the cost of infrastructure development and ensuring equitable access to 5G technology across urban and rural areas remain critical issues for governments and telecommunications providers.""",

    "Renewable Energy Adoption": """The adoption of renewable energy sources like solar, wind, hydroelectric, and geothermal power is a key strategy in the global effort to reduce greenhouse gas emissions and combat climate change. Technological advancements have significantly lowered the cost of renewable energy, making it increasingly competitive with fossil fuels. However, integrating these variable energy sources into existing power grids poses technical challenges, particularly in terms of energy storage and grid stability. Governments are implementing policies such as subsidies, feed-in tariffs, and carbon pricing to incentivize the transition. Public and private sector investments in battery technology, smart grids, and infrastructure are also critical to scaling up renewable energy adoption. Ultimately, transitioning to a low-carbon energy system is essential for a sustainable future.""",

    "Mental Health and Technology": """Technology is increasingly being leveraged to address mental health challenges through digital platforms, mobile applications, and wearable devices. These tools offer opportunities for early detection, self-management, and remote monitoring of conditions like anxiety, depression, and PTSD. AI-powered chatbots, mood tracking apps, and teletherapy have made mental health resources more accessible, especially in underserved areas. However, concerns persist regarding data privacy, the efficacy of self-guided interventions, and the potential overreliance on technology over human interaction. Ensuring clinical validation and maintaining ethical standards in digital mental health solutions are crucial. Moreover, digital divides and varying levels of digital literacy can limit the equitable distribution and effectiveness of these technologies.""",

    "Space Exploration": """Space exploration has entered a new era, characterized by increased participation from private companies and renewed interest from governments. Advances in reusable rocket technology, spearheaded by companies like SpaceX and Blue Origin, have significantly reduced the cost of space missions. Missions to the Moon, Mars, and beyond are being planned not just for scientific research but also for potential colonization and resource extraction. However, space exploration is not without challenges, including high financial costs, exposure to cosmic radiation, psychological effects of long-duration missions, and ethical questions about space colonization. International cooperation, sustainable practices, and robust safety protocols are essential to ensure responsible exploration and utilization of space.""",

    "NLP and Language Models": """Natural Language Processing (NLP) has evolved rapidly due to advancements in deep learning and the development of large-scale language models like BERT, GPT, and T5. These models can perform a wide range of tasks, including machine translation, summarization, question answering, and content generation. NLP is being integrated into industries such as customer service, healthcare, education, and legal analysis. Despite their capabilities, these models face several limitations, including the risk of generating biased or misleading outputs, high computational resource requirements, and lack of explainability. Researchers are working on improving model interpretability, reducing energy consumption, and creating more ethical and responsible AI systems to address these concerns and broaden the safe application of NLP technologies."""
}

style_options = {
    "Simple": "Explain the following concept in very simple and easy-to-understand language, avoiding technical jargon and using everyday analogies: ",
    
    "Technical": "Provide a detailed and technical summary of the following text, using domain-specific terms, referencing key processes, methodologies, or frameworks as relevant to an expert audience: ",
    
    "Creative": "Summarize the following text in a creative, engaging, and story-like manner. Feel free to use metaphors, imagery, or narrative tone to make the summary feel vivid and interesting: ",
    
    "Bullet Points": "Summarize the main ideas of the following text using concise bullet points. Each bullet should capture one core idea or argument, keeping clarity and completeness in mind: ",
    
    "For a 10-year-old": "Explain the following in a way that a curious 10-year-old would understand. Use very simple language, analogies a child might relate to, and keep it fun and educational: "
}



length_settings = {
    "Short": {"min": 200, "max": 300},
    "Medium": {"min": 500, "max": 800},
    "Long": {"min": 1000, "max": 2000}
}


st.title(" Hugging Face Research Paper Summarizer")

selected_text_key = st.selectbox(" Select a research topic:", list(sample_texts.keys()))
selected_style_key = st.selectbox(" Choose explanation style:", list(style_options.keys()))
selected_length_key = st.selectbox(" Select summary length:", list(length_settings.keys()))


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
            do_sample=False,
             no_repeat_ngram_size=3,
            early_stopping=True
        )[0]["summary_text"]

    st.subheader("📄 Summary")
    st.write(summary)
