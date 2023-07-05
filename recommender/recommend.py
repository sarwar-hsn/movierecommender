import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import os

# Load datasets
current_dir = os.path.dirname(os.path.abspath(__file__))
credit_file = os.path.join(current_dir, 'tmdb_5000_credits.csv')
movies_file = os.path.join(current_dir, 'tmdb_5000_movies.csv')

credit = pd.read_csv(credit_file)
movies = pd.read_csv(movies_file)


integrated = pd.concat([credit, movies], axis=1)
df = integrated[['original_title', 'id', 'cast', 'crew', 'budget', 'release_date', 'genres', 'keywords', 'original_language', 'overview', 'popularity', 'production_companies', 'production_countries', 'revenue', 'runtime', 'spoken_languages', 'vote_average']]

# Preprocess data
for column in ['genres', 'cast', 'keywords']:
    df[column] = df[column].fillna("[]").apply(eval).apply(lambda x: [i["name"] for i in x])
df.loc[:, 'crew'] = df['crew'].fillna("[]").apply(eval).apply(lambda x: [i["name"] for i in x if i["job"]=="Director"])
df.loc[:, 'text'] = df['genres'].apply(' '.join) + ' ' + df['keywords'].apply(' '.join) + ' ' + df['overview']
df = df.dropna()

# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["text"].fillna(''))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Load the model
model_path = os.path.join(current_dir, 'movie_recommender.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Define function to get recommendations
indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    # Fuzzy matching to correct misspelled titles
    title = process.extractOne(title, df['original_title'])[0]
    idx = indices[title]

    #if multiple indices returned and select the first one if so
    if type(idx) == pd.Series:
        idx = idx.iloc[0]

    sim_scores = list(enumerate(cosine_sim[idx].flatten())) # Flatten array here
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in sim_scores[1:11]] #returning first 10 movies
    return df[['original_title', 'id']].iloc[movie_indices]


def get_movie_by_id(movie_id):
    movie = df[df['id'] == movie_id]
    return movie.iloc[0].to_dict() if len(movie) > 0 else None
