# 📚 AI Book Recommendation System

An NLP-based Book Recommendation System built using **Python, Streamlit, TF-IDF, and Word2Vec**. The system collects book data from an API, preprocesses textual information, generates text embeddings, and recommends books based on user preferences.

---

## Features

* Fetches book data from the Google Books API
* Performs complete NLP preprocessing pipeline
* Stores processed data in CSV format
* Supports two recommendation techniques:

  * TF-IDF
  * Word2Vec
* Interactive Streamlit interface
* Returns recommended books with similarity scores
* Compares keyword-based and embedding-based retrieval approaches

---

## Project Workflow

```text
Google Books API
        ↓
Data Collection
        ↓
Data Cleaning
        ↓
Text Preprocessing
        ↓
Save Processed Data (CSV)
        ↓
TF-IDF Vectorization
        ↓
Word2Vec Embeddings
        ↓
Cosine Similarity
        ↓
Book Recommendations
```

---

## Data Collection

Book metadata was collected using the **Google Books API**.

The following information was extracted:

* Book Title
* Authors
* Categories
* Description
* Published Date

After collection, the data was stored in a CSV file for further processing.

---

## NLP Preprocessing

The book descriptions were processed using a complete NLP pipeline:

* Lowercasing
* Removing punctuation
* Removing special characters
* Tokenization
* Stopword removal
* Lemmatization

The final cleaned text was stored in a new column called:

```text
processed_text
```

The processed dataset was then saved as:

```text
books_data.csv
```

---

## Text Representation

### TF-IDF

TF-IDF converts text into numerical vectors based on word importance.

Advantages:

* Fast
* Lightweight
* Easy to implement

Limitation:

* Relies on keyword matching
* Does not understand semantic meaning

Example:

```text
AI ≠ Artificial Intelligence
ML ≠ Machine Learning
```

---

### Word2Vec

Word2Vec generates dense word embeddings that capture semantic relationships between words.

Advantages:

* Better semantic understanding than TF-IDF
* Similar words have similar vector representations

Example:

```text
King ≈ Queen
AI ≈ Machine Learning
```

---

## Recommendation Engine

The recommendation system uses **Cosine Similarity** to compare user queries with book representations.

### TF-IDF Recommendation

```text
User Query
      ↓
TF-IDF Vector
      ↓
Cosine Similarity
      ↓
Top Matching Books
```

### Word2Vec Recommendation

```text
User Query
      ↓
Word2Vec Embedding
      ↓
Cosine Similarity
      ↓
Top Matching Books
```

---

## Project Files

```text
Book-Recommendation-System/
│
├── app.py
├── books_data.csv
├── tfidf.pkl
├── tfidf_matrix.pkl
├── word2vec.model
├── tf_idf.ipynb
├── word2vec.ipynb
└── README.md
```

---

## Installation

```bash
pip install streamlit pandas numpy scikit-learn gensim nltk
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Example Queries

```text
machine learning
```

```text
deep learning neural networks
```

```text
history romance
```

```text
fantasy magic adventure
```

```text
psychology self improvement
```

---

## Future Improvements

* Sentence-BERT (SBERT)
* Semantic Search
* Hybrid Recommendation System
* Personalized Recommendations
* Book Cover Integration
* Author-Based Recommendations

---

## Learning Outcomes

This project helped in understanding:

* API Integration
* Data Collection
* NLP Preprocessing
* TF-IDF Vectorization
* Word2Vec Embeddings
* Cosine Similarity
* Recommendation Systems
* Streamlit Deployment

---

## Author

**Jainab Bee**

