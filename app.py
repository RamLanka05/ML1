import sentence_transformers
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Rigveda Vector Search", page_icon="🕉️", layout="centered")

@st.cache_data
def load_data():
    df = pd.read_parquet("rigveda_clean.parquet")
    mat = np.vstack(df["embedding"].values)
    return df, mat

@st.cache_resource
def load_model():
    model = sentence_transformers.SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return model

df, mat = load_data()
model = load_model()

with st.sidebar:
    st.header("About This Engine")
    st.markdown("""
    This tool uses a 384-dimensional **Large Language Model** to map the abstract concepts of the Rigveda. 
    
    Instead of relying on exact keyword matches, it computes the *cosine similarity* between your query and the text, returning verses based on pure meaning.
    """)

    st.divider()
    st.markdown("**Tech Stack:**\n* Hugging Face (`MiniLM-L12`)\n* Scikit-Learn\n* Pandas / Parquet\n* Streamlit")

st.title("🕉️ Rigveda Semantic Search")

c1, c2 = st.columns([4, 1])

with c1:
    query = st.text_input("Search Bar", label_visibility="collapsed", placeholder="e.g. 'cosmic order', 'sacrifice', 'divine knowledge'")
with c2:
    search_clicked = st.button("Search", use_container_width=True)


if search_clicked and query:
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, mat)
    top_indices = np.argsort(similarity_scores[0])[::-1][:10]
    top_results = df.iloc[top_indices].copy()
    top_results['score'] = similarity_scores[0][top_indices]
    
    st.divider()
    st.subheader("Top Semantic Matches")
    
    for index, row in top_results.iterrows():
        with st.container():
            meta_col1, meta_col2 = st.columns([3, 1])
            
            with meta_col1:
                st.markdown(f"### Mandala {row['mandala']}, Sukta {row['sukta']}, Verse {row['verse_num']}")
            with meta_col2:
                st.metric(label="Match Score", value=f"{row['score']:.4f}")

            st.markdown(f"**English:** {row['english_translation']}")
            st.markdown(f"**Sanskrit:** {row['sanskrit_verse']}")
            
            st.divider()