# Rigveda Semantic Search Engine

## Overview
This project is an end-to-end Machine Learning pipeline and interactive web application designed to explore the **Rigveda**, the oldest of the sacred Vedic Sanskrit texts. 

By transforming complex, deeply nested hierarchical data into a flat Parallel Corpus and passing it through a multilingual Large Language Model (LLM), this tool allows users to search ancient scriptures using modern **Semantic Vector Search**. Instead of relying on exact keyword matches, the engine understands the abstract meaning of a user's query (e.g., "the creation of the universe" or "chariots of war") and returns the most conceptually relevant Sanskrit verses and their English translations.

## The Problem: Beyond Keyword Search
The Rigveda is massive, ancient, and highly poetic. Traditional keyword search fails when exploring these texts because a user might search for "fire," while the translation uses "flame," "Agni," or "oblations." 

To solve this, the project uses **NLP Embeddings** to map the abstract concepts of the text into a 384-dimensional vector space, allowing for similarity matching based on *meaning* rather than exact phrasing.

## Tech Stack & Architecture
- **Frontend:** Streamlit (Cached data and model loading for instant UI response)
- **Machine Learning:** Hugging Face `SentenceTransformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector Math:** Scikit-Learn (`cosine_similarity`)
- **Data Engineering:** Pandas, NumPy
- **Storage:** Parquet (Compressed columnar storage bypassing standard CSV limits)

## Features
- **Semantic Retrieval Engine:** Computes cosine similarity between user queries and 10,000+ Vedic verses in real-time.
- **Robust Data Pipeline:** Parses irregular, nested dictionary structures (Mandala → Sukta → Verse) into a clean, machine-readable format.
- **Lightning-Fast UI:** Utilizes Streamlit's `@st.cache_resource` and `@st.cache_data` decorators to hold the 384-dimension LLM and the vector matrix in memory.
- **Parallel Text Alignment:** Aligns the raw Devanagari text, accented Vedic text (Svaras), and English translations for side-by-side analysis.

## Data Dictionary
The underlying dataset (`rigveda_clean.parquet`) contains the following structure:

| Column Name | Type | Description |
|-------------|------|-------------|
| `mandala` | String | The book number (1-10) of the Rigveda. |
| `sukta` | String | The hymn number within the Mandala. |
| `verse_num` | Int | The specific verse (Rik) number. |
| `sanskrit_verse` | String | Samhita text. Continuous recitation format. |
| `display_sanskrit` | String | Accented text containing Vedic Svaras. |
| `english_translation` | String | English translation of the verse. |
| `embedding` | Array[Float] | The 384-dimensional vector representation of the translation. |

## Setup & Installation
1. Clone the repository and navigate to the directory:
```bash
git clone [https://github.com/RamLanka05/Sanskrit-Analysis.git](https://github.com/RamLanka05/Sanskrit-Analysis.git)
cd Sanskrit-Analysis

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install streamlit pandas numpy scikit-learn sentence-transformers pyarrow
```

4. Run the application:

```bash
streamlit run app.py
```

## Roadmap & Future Work
With the core semantic engine and UI deployed, future iterations will focus on advanced exploratory data analysis:

1. Visual Data Mapping: Implement UMAP/t-SNE dimensionality reduction to create a 2D interactive scatter plot showing the semantic "clusters" of the Rigveda.

2. Linguistic Analysis: Implement Sandhi Splitting tools (e.g. CLTK) to split merged compound Sanskrit words.

3. Named Entity Recognition: Map the frequency and contextual sentiment of specific deities (Agni, Indra, Soma) across different Mandalas.


## Acknowledgments
Original raw data structure based on WisdomLib/Vedic textual archives.