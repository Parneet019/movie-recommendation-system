import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

movies = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "tmdb_5000_movies.csv")
)

credits = pd.read_csv(
    os.path.join(BASE_DIR, "dataset", "tmdb_5000_credits.csv")
)

print("Movies dataset:", movies.shape)
print("Credits dataset:", credits.shape)
movies = movies.merge(credits, on="title")

print("After merging:", movies.shape)
movies = movies[
    ["id","title", "overview", "genres", "keywords", "cast", "crew"]
]
def convert(obj):
    L = []

    for i in ast.literal_eval(obj):
        L.append(i["name"])

    return L
def convert3(obj):
    L = []

    counter = 0

    for i in ast.literal_eval(obj):

        if counter < 3:
            L.append(i["name"])

        counter += 1

    return L
def fetch_director(obj):
    L = []

    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])

    return L


movies["genres"] = movies["genres"].apply(convert)
print(movies["genres"].head())

movies["keywords"] = movies["keywords"].apply(convert)
print(movies["keywords"].head())

movies["cast"] = movies["cast"].apply(convert3)
print(movies["cast"].head())

movies["crew"] = movies["crew"].apply(fetch_director)
print(movies["crew"].head())

movies["tags"] = (
    movies["overview"].fillna("") + " " +
    movies["genres"].apply(lambda x: " ".join(x)) + " " +
    movies["keywords"].apply(lambda x: " ".join(x)) + " " +
    movies["cast"].apply(lambda x: " ".join(x)) + " " +
    movies["crew"].apply(lambda x: " ".join(x))
)
print(movies[["title", "tags"]].head())
movies = movies[["id","title", "tags"]]
movies["tags"] = movies["tags"].apply(lambda x: x.lower())

tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
vectors = tfidf.fit_transform(movies["tags"])
similarity = cosine_similarity(vectors)
print(similarity.shape)

movie_indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

def recommend(title):
    title = title.strip().lower()

    movie_indices_lower = {
        movie_title.lower(): index
        for movie_title, index in movie_indices.items()
    }

    if title not in movie_indices_lower:
        return []

    index = movie_indices_lower[title]

    distances = similarity[index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )

    recommendations = []

    for i in movie_list[1:6]:
        movie = movies.iloc[i[0]]

        recommendations.append({
            "title": movie["title"],
            "id": movie["id"]
        })

    return recommendations