# =======Import libraries=======
import spacy


# =======Global variables=======

# Read contents of 'movies.txt' into dictionary 'movies'
movies = {}
with open("movies.txt", "r") as f:
    for line in f:
        key, value = line[6], line[9:].replace("\n", "")
        movies[key] = value

# Save comparison review to string
comparison_review = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator."


# =======Define functions=======

# Function to compare comparison review with movie reviews in dictionary, receives comparison review as string, returns most similar movie
def recommend_movie(movie_enjoyed):

# Load model containing word vectors as will be relying on them to compute semantic similarity
    nlp = spacy.load('en_core_web_md')

    # Create Doc object from movie_enjoyed to use as model for comparison
    model_review = nlp(movie_enjoyed)

    # Initialise variables to hold highest similarity value and title of most similar movie
    highest_similarity = 0
    most_similar_movie = ""

    # Iterate through movies comparing semantic similarity with movie_enjoyed, save 'highest_similarity' value and 'most_similar_movie' title
    for title in movies:
        similarity = nlp(movies[title]).similarity(model_review)
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_movie = title
    
    return most_similar_movie


# ============ Main ============

# Output recommendation by calling function recommend_movie
print("\nYou enjoyed watching Planet Hulk: we think you'll enjoy Movie", recommend_movie(comparison_review), "\n")