import pandas as pd

movies = pd.read_csv("data/tmdb_5000_movies.csv")
credits = pd.read_csv("data/tmdb_5000_credits.csv")

print("MOVIE COLUMNS:")
print(movies.columns.tolist())

print("\nCREDIT COLUMNS:")
print(credits.columns.tolist())

print("\nMOVIE DATA:")
print(movies[["title", "genres", "overview", "keywords", "vote_average"]].head())