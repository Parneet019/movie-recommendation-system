from flask import Flask, render_template, request
from portfolio.movie_recommender import recommend

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def get_recommendations():

    movie = request.form["movie"]

    recommendations = recommend(movie)

    if not recommendations:
        return render_template(
            "results.html",
            movie=movie,
            recommendations=[],
            not_found=True
        )

    return render_template(
        "results.html",
        movie=movie,
        recommendations=recommendations,
        not_found=False
    )


if __name__ == "__main__":
    app.run(debug=True)