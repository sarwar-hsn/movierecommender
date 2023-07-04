import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# Load datasets
credit = pd.read_csv('recommender/tmdb_5000_credits.csv')
movies = pd.read_csv('recommender/tmdb_5000_movies.csv')

integrated = pd.concat([credit,movies], axis = 1)
df = integrated[['original_title','cast','crew','budget','release_date','genres','keywords','original_language','overview','popularity','production_companies','production_countries','revenue','runtime','spoken_languages','vote_average']]

# Preprocess data
for column in ['genres', 'cast', 'keywords']:
    df[column] = df[column].fillna("[]").apply(eval).apply(lambda x: [i["name"] for i in x])
df['crew'] = df['crew'].fillna("[]").apply(eval).apply(lambda x: [i["name"] for i in x if i["job"]=="Director"])
df['text'] = df['genres'].apply(' '.join) + ' ' + df['keywords'].apply(' '.join) + ' ' + df['overview']
df = df.dropna()


# Compute TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["text"].fillna(''))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(cosine_sim.shape[0],), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(cosine_sim.shape[0], activation='softmax')
])

# Compile and fit the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(cosine_sim, cosine_sim, epochs=5, batch_size=100)

# Save the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("movie_recommender.tflite", "wb") as f:
    f.write(tflite_model)
