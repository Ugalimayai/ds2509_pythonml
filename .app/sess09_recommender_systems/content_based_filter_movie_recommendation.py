"""
======================================================================================================================
Python script to demonstrate Content Based Filtering to recommend similar movies to users
on streaming platforms like Amazon Prime, Hulu or Netflix
======================================================================================================================


Requirements
-------------------------
pip install pandas, scikit-learn

Author: Karanja Mwaniki
Date: 26 September 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================================================================================
# SECTION 1: Create a sample dataset of movies
# ======================================================================================================================
# NB: For a real-world application the data would come from a DB, CSV or JSON file
data = {
    "title": [
        "The Matrix",
        "John Wick",
        "Inception",
        "Interstellar",
        "The Dark Knight",
        "Avengers: Endgame",
        "Titanic",
        "The Notebook",
        "Gladiator",
        "The Shawshank Redemption",
        "Forrest Gump",
        "The Godfather",
        "Pulp Fiction",
        "The Lion King",
        "Jurassic Park",
        "The Silence of the Lambs",
        "Frozen",
        "The Social Network",
        "Mad Max: Fury Road",
        "La La Land",
        "Get Out",
        "The Wolf of Wall Street",
        "Harry Potter and the Sorcerer's Stone"
    ],
    "genre": [
        "Action Sci-Fi",
        "Action Thriller",
        "Sci-Fi Thriller",
        "Sci-Fi Drama",
        "Action Crime",
        "Action Superhero",
        "Romance Drama",
        "Romance Drama",
        "Action Drama",
        "Drama",
        "Drama Romance",
        "Crime Drama",
        "Crime Drama",
        "Animation Family",
        "Adventure Sci-Fi",
        "Thriller Crime",
        "Animation Musical",
        "Drama Biography",
        "Action Adventure",
        "Romance Musical",
        "Horror Thriller",
        "Biography Comedy Drama",
        "Fantasy Adventure"
    ],
    "keywords": [
        "virtual reality hacker AI",
        "assassin revenge crime",
        "dream subconscious mind",
        "space time blackhole",
        "joker vigilante batman",
        "superheroes time travel",
        "ship iceberg love",
        "love relationship drama",
        "roman empire revenge gladiator",
        "prison escape hope friendship",
        "life journey love destiny",
        "mafia family power crime",
        "nonlinear crime violence",
        "lion king africa destiny",
        "dinosaurs park science chaos",
        "serial killer investigation FBI",
        "ice powers sister love",
        "facebook startup ambition betrayal",
        "postapocalyptic desert survival",
        "music love dreams hollywood",
        "racism hypnosis horror",
        "stock market greed corruption",
        "wizard magic school friendship"
    ],
    "overview": [
        "A hacker discovers reality is a simulation.",
        "An ex-hitman seeks revenge.",
        "A thief enters dreams to steal secrets.",
        "Explorers travel through a wormhole in space.",
        "Batman faces the Joker in Gotham.",
        "Heroes unite to reverse a catastrophe.",
        "A love story aboard a doomed ship.",
        "A romantic story about enduring love.",
        "A betrayed general seeks revenge in ancient Rome.",
        "Two imprisoned men bond over years and find redemption.",
        "A man's extraordinary life journey unfolds through decades.",
        "The aging patriarch transfers control of his empire to his son.",
        "Interwoven stories of crime in Los Angeles.",
        "A lion prince flees and returns to reclaim his kingdom.",
        "Scientists recreate dinosaurs in a theme park gone wrong.",
        "An FBI trainee hunts a brilliant but twisted serial killer.",
        "A princess with ice powers struggles to control her abilities.",
        "The rise of Facebook and the conflicts behind it.",
        "Survivors race across a desert in a post-apocalyptic world.",
        "Two artists pursue love and dreams in Los Angeles.",
        "A young man uncovers a disturbing secret about his girlfriend's family.",
        "A stockbroker rises and falls amid greed and excess.",
        "A young wizard begins his journey at a magical school."
    ]
}

# Convert to PD DataFrame
df = pd.DataFrame(data)

# ======================================================================================================================
# SECTION 2: Combine the features into one text column
# ======================================================================================================================
# This helps the algorithm treat all features as one
df["combined_features"] = (
    df["genre"] + " " +
    df["keywords"] + " " +
    df["overview"]
)

# ======================================================================================================================
# SECTION 3: Convert text to numerical vectors (TF-IDF)
# ======================================================================================================================
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

# ======================================================================================================================
# SECTION 4: Compute cosine similarity
# ======================================================================================================================
# This measures/computes how similar each movie is to others
similarity_matrix = cosine_similarity(tfidf_matrix)

# ======================================================================================================================
# SECTION 5: Import the required modules
# ======================================================================================================================
def recommend_movies(movie_title, number_of_recommendations = 3):

    # check if the movei exists
    if movie_title not in df["title"].values:
        return f"{movie_title} was not found in the dataset."

    # Get the index of the movie
    movie_index = df[df["title"] == movie_title].index[0]

    # Get the similarity scores
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    # Arrange movies based on similarity score (highest first)
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # skip the first movie (if it's the same as what we're searching recommedations for)
    sorted_movies = sorted_movies[1:]


    # get/fetch the top n
    recommended = []
    for n in sorted_movies[:number_of_recommendations]:
        recommended.append(df.iloc[n[0]]["title"])
    return recommended

# ======================================================================================================================
# SECTION 6: Find similar movies
# ======================================================================================================================
if __name__ == "__main__":
    movie = "Gladiator"
    print(f"Movies similar to {movie}:")
    recommendations = recommend_movies(movie, number_of_recommendations = 5)

    for idx, rec, in enumerate(recommendations, start=1):
        print(f"{idx}: {rec}")
