import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_data():
    try:
        movies_df = pd.read_csv("movies.csv")
    except FileNotFoundError:
        st.error("movies.csv file not found. Please keep movies.csv in the same folder as app.py.")
        st.stop()

    movies_df = movies_df.dropna(subset=["title", "genres"]).copy()
    movies_df["genres"] = movies_df["genres"].replace("(no genres listed)", "")
    movies_df["clean_genres"] = movies_df["genres"].str.replace("|", " ", regex=False)

    return movies_df


@st.cache_resource
def build_similarity_matrix(clean_genres):
    vectorizer = CountVectorizer()
    genre_matrix = vectorizer.fit_transform(clean_genres)
    similarity_matrix = cosine_similarity(genre_matrix)

    return similarity_matrix


def recommend_movies(movie_title, movies_df, similarity_matrix, movie_indices, top_n=10):
    if movie_title not in movie_indices:
        return pd.DataFrame()

    movie_index = movie_indices[movie_title]

    if isinstance(movie_index, pd.Series):
        movie_index = movie_index.iloc[0]

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:top_n + 1]

    movie_indexes = [index for index, score in similarity_scores]
    scores = [round(score, 2) for index, score in similarity_scores]

    recommendations = movies_df.iloc[movie_indexes][["title", "genres"]].copy()
    recommendations["similarity_score"] = scores
    recommendations = recommendations.rename(
        columns={
            "title": "Movie Title",
            "genres": "Genres",
            "similarity_score": "Similarity Score"
        }
    )

    recommendations["Genres"] = recommendations["Genres"].str.replace("|", " | ", regex=False)

    return recommendations.reset_index(drop=True)


movies = load_data()
similarity = build_similarity_matrix(movies["clean_genres"])
movie_indices = pd.Series(movies.index, index=movies["title"])


st.title("🎬 Movie Recommendation System")
st.write(
    "A content-based movie recommendation system using the MovieLens dataset, "
    "CountVectorizer, and Cosine Similarity."
)

st.markdown("---")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Select Movie")

    search_query = st.text_input("Search movie title", placeholder="Example: Batman, Toy Story, Avengers")

    if search_query:
        filtered_movies = movies[
            movies["title"].str.contains(search_query, case=False, na=False)
        ]["title"].sort_values().tolist()
    else:
        filtered_movies = movies["title"].sort_values().tolist()

    if filtered_movies:
        selected_movie = st.selectbox("Choose a movie", filtered_movies)
        top_n = st.slider("Number of recommendations", 5, 20, 10)

        recommend_button = st.button("Recommend Movies", use_container_width=True)
    else:
        st.warning("No movies found. Try another search keyword.")
        selected_movie = None
        recommend_button = False

with right_col:
    st.subheader("Recommendations")

    if recommend_button and selected_movie:
        recommendations = recommend_movies(
            selected_movie,
            movies,
            similarity,
            movie_indices,
            top_n
        )

        if recommendations.empty:
            st.warning("No recommendations found.")
        else:
            st.success(f"Showing movies similar to: {selected_movie}")
            st.dataframe(recommendations, use_container_width=True)
    else:
        st.info("Search and select a movie, then click Recommend Movies.")


st.markdown("---")

st.subheader("About This Project")
st.write(
    """
    This project is a simple content-based movie recommendation system.
    It recommends movies based on genre similarity.

    The system uses the MovieLens dataset. Movie genres are converted into numerical
    vectors using CountVectorizer. Cosine Similarity is then used to calculate how
    similar one movie is to another.
    """
)

st.subheader("Technologies Used")
st.write(
    """
    - Python
    - Pandas
    - Scikit-learn
    - Streamlit
    - MovieLens Dataset
    """
)