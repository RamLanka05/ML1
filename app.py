import sentence_transformers
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Rigveda Vector Search", page_icon="🕉️", layout="wide")

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
    st.markdown("""
    **Tech Stack:**
    - [Sentence-Transformers `paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) — model card
    - [Sentence-Transformers docs](https://www.sbert.net/) — library docs and usage
    - [Scikit-Learn](https://scikit-learn.org/stable/) — ML utilities & cosine similarity
    - [Pandas](https://pandas.pydata.org/) — dataframes (Parquet support)
    - [Apache Parquet](https://parquet.apache.org/) — columnar storage format
    - [Streamlit](https://docs.streamlit.io/) — app framework
    """)

st.title("🕉️ Rigveda Semantic Search")

c1, c2 = st.columns([4, 1])

with c1:
    query = st.text_input("Search Bar", label_visibility="collapsed", placeholder="e.g. 'cosmic order', 'sacrifice', 'divine knowledge'")
with c2:
    search_clicked = st.button("Search", width="stretch")

# --- INITIALIZE MAIN DASHBOARD COLUMNS PERMANENTLY ---
left_col, right_col = st.columns([2, 3])

# Create a clean baseline copy for plotting layers
df_plot = df.copy()
df_plot['is_match'] = 'Background Corpus'

has_searched = False
top_results = None

# Calculate semantic layers if search conditions are met
if (search_clicked or query) and query.strip() != "":
    query_embedding = model.encode([query])
    similarity_scores = cosine_similarity(query_embedding, mat)
    top_indices = np.argsort(similarity_scores[0])[::-1][:10]
    
    top_results = df.iloc[top_indices].copy()
    top_results['score'] = similarity_scores[0][top_indices]
    
    # Mark the high-contrast visualization target elements
    df_plot.loc[top_indices, 'is_match'] = "Top Results"
    has_searched = True

# --- LEFT COLUMN: RESULTS PANELS ---
with left_col:
    st.divider()
    st.subheader("Top Semantic Matches")
    
    if has_searched:
        with st.container(height=600):
            for index, row in top_results.iterrows():
                with st.container():
                    meta_col1, meta_col2 = st.columns([3, 1])

                    with meta_col1:
                        st.markdown(f"### {row['mandala']}, {row['sukta']}, Verse {row['verse_num']}")
                    with meta_col2:
                        st.metric(label="Match Score", value=f"{row['score']:.4f}")

                    st.markdown(f"**English:** {row['english_translation']}")
                    st.markdown(f"**Sanskrit:** {row['sanskrit_verse']}")
                    st.divider()
    else:
        st.info("💡 Type an abstract concept or a deity's name above and hit enter to view semantic matches and map the cluster network!")

# --- RIGHT COLUMN: PERMANENT CLUSTER MAP ---
with right_col:
    st.subheader("🗺️ Vedic Concept Space Map (t-SNE Clusters)")
    st.markdown("""
    * **How to read this map:** Each dot represents an entire verse. 
    * **Navigation:** Drag to box-zoom. Double-click to reset. Hover for verse details.
    """)

    # --- THE RICH TOOLTIP UPGRADE ---
    # We dynamically build the HTML tooltip depending on if we have similarity scores to show
    if has_searched:
        df_plot['hover_preview'] = (
            "<b>" + df_plot['mandala'].astype(str) + 
            ", " + df_plot['sukta'].astype(str) + 
            ", V" + df_plot['verse_num'].astype(str) + "</b><br>" +
            "<i>Match Score: " + df_plot['similarity_score'].round(4).astype(str) + "</i><br><br>" +
            df_plot['english_translation'].str.wrap(60).str.replace('\n', '<br>')
        )
    else:
        df_plot['hover_preview'] = (
            "<b>" + df_plot['mandala'].astype(str) + 
            ", " + df_plot['sukta'].astype(str) + 
            ", V" + df_plot['verse_num'].astype(str) + "</b><br><br>" +
            df_plot['english_translation'].str.wrap(60).str.replace('\n', '<br>')
        )

    # Render out the complete scatter matrix canvas space
    fig = px.scatter(
        df_plot.sort_values(by='is_match'), 
        x='x_coord',
        y='y_coord',
        color='is_match',
        color_discrete_map={
            'Background Corpus': 'rgba(150, 150, 150, 0.25)', 
            'Top Results': '#FFD700'                            
        },
        custom_data=['hover_preview'], 
        labels={'is_match': 'Layer'},
        template='plotly_dark' 
    )

    # 1. Restore background dots to a legible but clean size (6px)
    fig.update_traces(
        marker=dict(size=6), 
        selector=dict(name='Background Corpus'),
        showlegend=False
    )
    
    # 2. Make the Top Results pop by turning them into large stars
    if has_searched:
        fig.update_traces(
            marker=dict(size=18, symbol='star', line=dict(width=1, color='black')), 
            selector=dict(name='Top Results')
        )

    # 3. Clean up Layout: Anchor legend above the box and increase top margin
    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, b=10, t=50), # Increased 't' to 50 to give the legend breathing room
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        legend=dict(
            orientation="h", 
            y=1.01,           # Push slightly above the top border
            yanchor="bottom", # Anchor from the bottom of the legend so it expands upward, not downward
            x=0.01,           # Slight indent from the left border
            xanchor="left", 
            title=""
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, title="",
            showline=True, mirror=True, linecolor="rgba(255, 255, 255, 0.2)", linewidth=1
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, title="",
            showline=True, mirror=True, linecolor="rgba(255, 255, 255, 0.2)", linewidth=1
        ),
        dragmode="zoom" 
    )
    
    # Point exactly to index [0] where our rich HTML string is bundled
    fig.update_traces(hovertemplate="%{customdata[0]}<extra></extra>")

    st.plotly_chart(fig, use_container_width=True)