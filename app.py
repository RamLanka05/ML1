import sentence_transformers
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    
    st.subheader("🗺️ Vedic Concept Space Map (t-SNE Clusters)")
    st.markdown("""
    * **How to read this map:** Each dot represents an entire verse. Verses that sit close together share an abstract semantic or philosophical theme. 
    * **Highlighted Dots:** The bright yellow/gold stars represent the top matches to your search query.
    """)

    # Setup plotting layers to prevent high-dimensional frontend lag
    df_plot = df.copy()
    df_plot['similarity_score'] = similarity_scores[0]
    df_plot['is_match'] = 'Background Corpus'
    df_plot.loc[top_indices, 'is_match'] = f"Top Results"

    # Add a clean text wrap preview for the interactive hover popup
    df_plot['hover_preview'] = (
        "M" + df_plot['mandala'].astype(str) + 
        ", S" + df_plot['sukta'].astype(str) + 
        ", V" + df_plot['verse_num'].astype(str) + "<br>" +
        df_plot['english_translation'].str.wrap(50).str.replace('\n', '<br>')
    )

    # Render the scatter plot matrix
    fig = px.scatter(
        df_plot.sort_values(by='is_match'), # Sort to guarantee matches draw on top layer
        x='x_coord',
        y='y_coord',
        color='is_match',
        color_discrete_map={
            'Background Corpus': 'rgba(150, 150, 150, 0.25)', 
            'Top Results': '#FFD700'                            
        },
        size=df_plot['is_match'].apply(lambda x: 14 if x != 'Background Corpus' else 4),
        hover_data={'x_coord': False, 'y_coord': False, 'is_match': False, 'hover_preview': True},
        labels={'is_match': 'Layer'},
        template='plotly_dark' 
    )

    # Polish chart canvas padding, boundaries, and alignment parameters
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, b=0, t=10),
        legend=dict(orientation="h", y=1.02, x=0, xanchor="left"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="")
    )
    
    # Isolate tooltip strings to target hover metadata blocks cleanly
    fig.update_traces(hovertemplate="%{customdata[2]}<extra></extra>")

    st.plotly_chart(fig, use_container_width=True)