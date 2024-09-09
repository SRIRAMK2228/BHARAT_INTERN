from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)
# Load the datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print("Movies DataFrame Columns before merge:", movies.columns)
print("Credits DataFrame Columns before merge:", credits.columns)
movies = movies.merge(credits, left_on='id', right_on='movie_id')
print("Movies DataFrame Columns after merge:", movies.columns)
movies = movies.rename(columns={'title_x': 'title'})  # Adjust if necessary
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

def combine_features(row):
    return f"{row['overview']} {row['genres']} {row['keywords']} {row['cast']} {row['crew']}"
movies['combined_features'] = movies.apply(combine_features, axis=1)
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(movies['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return movies[movies.index == index]["title"].values[0]

def get_index_from_title(title):
    title = title.lower().strip()
    for i, t in enumerate(movies['title']):
        if t.lower().strip() == title:
            return i
    return None  
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/recommend', methods=['POST'])
def recommend():
    movie_user_likes = request.form['movie']
    movie_index = get_index_from_title(movie_user_likes)
    if movie_index is not None:
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:11]

        recommendations = []
        for element in sorted_similar_movies:
            recommendations.append(get_title_from_index(element[0]))
        return render_template('recommend.html', recommendations=recommendations, movie_user_likes=movie_user_likes)
    else:
        return render_template('index.html', error=f"No match found for '{movie_user_likes}'.")
if __name__ == '__main__':
    app.run(debug=True)
