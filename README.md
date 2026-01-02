# Sanskrit ML Analysis

## Overview
This project implements a data processing pipeline to parse, structure, and analyze ancient Sanskrit texts, including the **Rigveda**, the oldest of the sacred Vedic Sanskrit texts.

The core script transforms complex, deeply nested hierarchical data (organized by *Mandala* → *Sukta* → *Verse*) into a flat, machine-readable **Pandas DataFrame**. This structure serves as a foundational "Parallel Corpus" for downstream Natural Language Processing (NLP) and Digital Humanities research, enabling simultaneous analysis of the original Sanskrit (phonetics/meter) and English translations (semantics).

## The Data Problem
The raw data exists in a nested dictionary format where structural inconsistencies (e.g., varying depths for `samhita` vs `padapatha` text) make direct analysis difficult.

**Input Structure:**
`Mandala -> Sukta -> Verse -> {Nested Dictionary of Devanagari, Transliteration, Translations}`

**Output Structure:**
A clean, tabular DataFrame where every row represents a single *Rik* (verse) with aligned attributes.

## Features
- **Robust Parsing:** Handles exceptions and irregular schema structures (e.g., missing keys for specific verses).
- **Text Normalization:** Extracts both:
  - `sanskrit_verse`: Raw Devanagari text (suitable for string matching).
  - `display_sanskrit`: Accented Vedic text with Svaras (suitable for phonetic/prosody analysis).
- **Parallel Text Alignment:** Aligns the Sanskrit source with English translations for bitext analysis.

## Prerequisites
* Python 3.x
* Pandas

```bash
pip install pandas
```

## Data Dictionary
The resulting DataFrame (cleaned_df) contains the following columns:

| Column Name | Type | Description |
|-------------|------|-------------|
| mandala | String | The book number (1-10) of the Rigveda. |
| sukta | String | The hymn number within the Mandala. |
| verse_num | Int | The specific verse (Rik) number. |
| sanskrit_verse | String | Samhita text. Continuous recitation format. Best for general text processing. |
| display_sanskrit | String | Accented text. Contains Vedic accents (Svaras). Essential for analyzing meter and chanting intonation. |
| english_translation | String | English translation of the verse. Used for semantic analysis and topic modeling. |

## Usage
The extraction logic iterates through the source dictionary to flatten the structure:

```python
# (Simplified logic)
import pandas as pd

# Iterate through Mandalas and Suktas
# Extract Samhita and Translation
# Handle KeyErrors via try/except blocks

print(cleaned_df.head())
```

## Roadmap & Future Work
This project is currently in the Data Engineering phase. The next steps focus on NLP and Exploratory Data Analysis (EDA):

1. Linguistic Analysis (Sanskrit)
   - Sandhi Splitting: Implement tools (e.g., CLTK) to split merged compound words.
   - Meter Identification: Algorithmic counting of syllables to classify verses by meter (Gayatri, Tristubh, etc.).

2. Semantic Analysis (English)
   - Topic Modeling: Use LDA to cluster verses by theme (Cosmology, Ritual, Dialogue).
   - Named Entity Recognition: Map the frequency of deities (Agni, Indra, Soma) across different Mandalas.

3. Parallel Corpus Research
   - Investigating correlations between semantic topics and phonetic structures (e.g., Do verses about 'War' utilize specific meters?).


## Acknowledgments
Data structure based on WisdomLib/Vedic textual archives.