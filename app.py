import pickle
import pandas as pd
import numpy as np
import streamlit as st

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("books_data.csv")

if "processed_text" not in df.columns:
    df["description"] = df["description"].fillna("").astype(str)
    df["categories"] = df["categories"].fillna("").astype(str)
    df["processed_text"] = df["description"] + " " + df["categories"]

# TF-IDF
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

# Word2Vec
w2v_model = Word2Vec.load("word2vec.model")

def get_document_vector(text):

    words = str(text).lower().split()

    vectors = []

    for word in words:
        if word in w2v_model.wv:
            vectors.append(w2v_model.wv[word])

    if len(vectors) == 0:
        return np.zeros(w2v_model.vector_size)

    return np.mean(vectors, axis=0)


book_vectors = np.array([
    get_document_vector(text)
    for text in df["processed_text"]
])


def recommend_tfidf(query, top_n=5):

    query_vector = tfidf.transform([query])

    similarity_scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    results = pd.DataFrame({
        "Title": df.iloc[top_indices]["title"].values,
        "Author": df.iloc[top_indices]["authors"].values,
        "Score": similarity_scores[top_indices]
    })

    results["Score"] = results["Score"].round(4)

    return results


def recommend_word2vec(query, top_n=5):

    query_vector = get_document_vector(query)

    similarity_scores = cosine_similarity(
        [query_vector],
        book_vectors
    ).flatten()

    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    results = pd.DataFrame({
        "Title": df.iloc[top_indices]["title"].values,
        "Author": df.iloc[top_indices]["authors"].values,
        "Score": similarity_scores[top_indices]
    })

    results["Score"] = results["Score"].round(4)

    return results

st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Book Recommendation System")

st.write(
    """
    Enter your interests and get personalized book recommendations.
    Compare TF-IDF and Word2Vec recommendation techniques.
    """
)

method = st.selectbox(
    "Select Recommendation Method",
    [
        "TF-IDF",
        "Word2Vec"
    ]
)

user_input = st.text_area(
    "Describe the type of book you want",
    placeholder="machine learning, deep learning, romance history, fantasy magic..."
)

top_n = st.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=20,
    value=5
)

if st.button("Recommend Books"):

    if user_input.strip() == "":
        st.warning("Please enter a book description.")
    else:

        if method == "TF-IDF":
            recommendations = recommend_tfidf(
                user_input,
                top_n
            )
        else:
            recommendations = recommend_word2vec(
                user_input,
                top_n
            )

        st.success(
            f"Top {top_n} recommendations using {method}"
        )

        st.dataframe(recommendations)
