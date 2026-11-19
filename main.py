import pandas as pd
import ast
from dotenv import load_dotenv
import os
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

# Load environment variables
load_dotenv()

# Get TMDB API key
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def get_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    params = {
        "api_key": TMDB_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    return ""
# Load datasets
movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

# Select required columns
movies = movies[
    ["id", "title", "genres", "keywords", "overview", "vote_average"]
]

credits = credits[
    ["movie_id", "cast", "crew"]
]

# Merge datasets
movies = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)


# Convert JSON-like data into names
def convert(obj):
    result = []

    for item in ast.literal_eval(obj):
        result.append(item["name"])

    return result


# Extract genres
movies["genres"] = movies["genres"].apply(convert)

# Extract keywords
movies["keywords"] = movies["keywords"].apply(convert)


# Extract top 3 cast members
def get_cast(obj):
    result = []

    for item in ast.literal_eval(obj)[:3]:
        result.append(item["name"])

    return result


movies["cast"] = movies["cast"].apply(get_cast)


# Extract director
def get_director(obj):
    result = []

    for item in ast.literal_eval(obj):
        if item["job"] == "Director":
            result.append(item["name"])
            break

    return result


movies["crew"] = movies["crew"].apply(get_director)


# Display the result
print("\nMovie:", movies.iloc[0]["title"])
print("Genres:", movies.iloc[0]["genres"])
print("Keywords:", movies.iloc[0]["keywords"])
print("Cast:", movies.iloc[0]["cast"])
print("Director:", movies.iloc[0]["crew"])


# Fill missing overview values
movies["overview"] = movies["overview"].fillna("")

# Combine all important features into one column
movies["tags"] = (
    movies["genres"].apply(lambda x: " ".join(x))
    + " "
    + movies["keywords"].apply(lambda x: " ".join(x))
    + " "
    + movies["overview"]
    + " "
    + movies["cast"].apply(lambda x: " ".join(x))
    + " "
    + movies["crew"].apply(lambda x: " ".join(x))
)

# Display the tags for the first movie
print("\nTags for", movies.iloc[0]["title"])
print(movies.iloc[0]["tags"])

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

# Convert movie tags into numerical vectors
tfidf_matrix = tfidf.fit_transform(movies["tags"])

print("\nTF-IDF matrix shape:")
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

# Calculate similarity between all movies
similarity = cosine_similarity(tfidf_matrix)

print("\nSimilarity matrix shape:")
print(similarity.shape)

# Create movie recommendation function
def recommend(movie_title):
    # Find the movie index
    movie_index = movies[movies["title"] == movie_title].index[0]

    # Get similarity scores for this movie
    distances = similarity[movie_index]

    # Sort movies by similarity score
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    # Display top 5 similar movies
    print("\nMovies similar to", movie_title, ":")

    for i in movie_list[1:6]:
        print(movies.iloc[i[0]]["title"])


# Test the recommendation system
recommend("Avatar")

# Create Flask application

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    error = ""

    if request.method == "POST":

        movie_title = request.form["movie"].strip()

        matches = movies[
            movies["title"].str.lower() == movie_title.lower()
        ]

        if matches.empty:

            error = "Movie not found. Please try another movie."

        else:

            movie_index = matches.index[0]

            distances = similarity[movie_index]

            movie_list = sorted(
                list(enumerate(distances)),
                reverse=True,
                key=lambda x: x[1]
            )

            for i in movie_list[1:6]:

                movie = movies.iloc[i[0]]

                poster = get_poster(movie["id"])

                recommendations.append({
                    "title": movie["title"],
                    "id": movie["id"],
                    "poster": poster,
                    "rating": movie["vote_average"],
                    "overview": movie["overview"],
                    "genres": ", ".join(movie["genres"])
                })

    return render_template(
        "index.html",
        recommendations=recommendations,
        error=error
    )


if __name__ == "__main__":

    print("TMDB API key loaded:", bool(TMDB_API_KEY))

    app.run(debug=True)