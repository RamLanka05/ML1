import sentence_transformers
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.title("Sanskrit ML UI")

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

query = st.text_input("Enter a concept to search the Rigveda")
search_clicked = st.button("Search")


if search_clicked and query:
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, mat)
    top_indices = np.argsort(similarity_scores[0])[::-1][:5]
    top_results = df.iloc[top_indices].copy()
    top_results['score'] = similarity_scores[0][top_indices]
    
    st.write("Top 5 results:")
    for index, row in top_results.iterrows():
        # 1. Print the citation in bold using the exact column names
        st.markdown(f"**{row['mandala']}, {row['sukta']}, Verse {row['verse_num']}**")
        
        # 2. Print the score
        st.write(f"Match Score: {row['score']:.4f}")
        
        # 3. Print the actual text columns
        st.write(row['english_translation'])
        st.write(row['sanskrit_verse'])
        
        # 4. Draw a clean horizontal line
        st.divider()