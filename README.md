# Movie Recommendation System

A simple content-based movie recommendation system built using the MovieLens dataset.

## Project Description

This project recommends movies based on genre similarity. It uses the MovieLens dataset and applies content-based filtering using CountVectorizer and Cosine Similarity.

## Live Demo

https://movie-recommendation-system-04.streamlit.app/

## Features

- Search movies by title
- Select a movie from search results
- Recommend similar movies
- Adjustable number of recommendations
- Simple Streamlit web interface

## Dataset

Dataset used: MovieLens Latest Small Dataset

Main file used:

- movies.csv

Other dataset files included:

- ratings.csv
- tags.csv
- links.csv

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- MovieLens Dataset

## How It Works

1. Load the MovieLens movies dataset.
2. Clean the genres column.
3. Convert genres into numerical vectors using CountVectorizer.
4. Calculate similarity between movies using Cosine Similarity.
5. Recommend movies with the highest similarity scores.
