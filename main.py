import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from transformers import pipeline, GPT2TokenizerFast
import os

# Load dataset and model
dataset = load_dataset("Pablinho/movies-dataset")["train"]
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
generator = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=50,  # Set max_new_tokens instead of max_length to control only the generated tokens
    num_return_sequences=1,
    pad_token_id=tokenizer.eos_token_id  # Ensure sentences end smoothly
)

# Custom CSS for background and text colors
st.markdown(
    """
    <style>
    .stApp {
        background-color: #BFECFF;
    }
    h1, h2, h3, h4, h5, h6, p, .stButton>button {
        color: #EF5A6F;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Preprocess and cache embeddings
@st.cache_data
def preprocess_data(_dataset):
    _dataset = _dataset.filter(
        lambda x: x["Title"] is not None and x["Overview"] is not None and x["Genre"] is not None)
    _dataset = _dataset.map(lambda x: {"Overview": x["Overview"].lower().strip()})
    return _dataset


dataset = preprocess_data(dataset)

# Load or generate embeddings
embeddings_path = "data/movie_embeddings.npz"
if os.path.exists(embeddings_path):
    embeddings = np.load(embeddings_path)['embeddings']
else:
    embeddings = model.encode([movie['Overview'] for movie in dataset])
    np.savez(embeddings_path, embeddings=embeddings)


# Adjust mood based on genre
def adjust_mood_for_genre(mood, genre):
    if "Horror" in genre or "Drama" in genre:
        return "engaging and empowering" if "happy" in mood else mood
    return mood


# Recommendation function with contextual justification
def recommend_with_explanations(favorite_movies=None, mood_query=None, num_recommendations=5):
    recommendations = []
    total_tokens_used = 0

    # Simplify mood description if necessary
    simplified_mood = "happy" if mood_query else mood_query
    user_profile = np.mean(model.encode(favorite_movies) if favorite_movies else model.encode([simplified_mood]),
                           axis=0).reshape(1, -1)
    similarities = cosine_similarity(user_profile, embeddings).flatten()
    top_indices = similarities.argsort()[-num_recommendations:][::-1]

    for idx in top_indices:
        movie = dataset[int(idx)]
        movie_title = movie.get("Title", "N/A")
        movie_overview = movie.get("Overview", "No overview available")
        movie_genre = movie.get("Genre", "Genre not specified")

        # Adjust mood based on genre
        adjusted_mood = adjust_mood_for_genre(simplified_mood, movie_genre)

        # Construct a prompt that uses both the user's mood/preferences and movie overview for justification
        prompt = (
            f"The user is in the mood for something {adjusted_mood} and may enjoy movies like {', '.join(favorite_movies) if favorite_movies else 'other similar movies'}. "
            f"Explain why the movie '{movie_title}' would be a good recommendation based on the following overview: '{movie_overview}'. "
            f"Focus on themes of friendship, resilience, or personal growth as they align with an {adjusted_mood} mood.")

        # Generate the explanation based on this contextual prompt
        explanation = generator(prompt)[0]["generated_text"]

        # Limit explanation to the first sentence for coherence
        explanation = explanation.split('.')[0] + '.' if '.' in explanation else explanation

        # Calculate tokens used
        tokens_used = len(tokenizer.encode(prompt + explanation))
        total_tokens_used += tokens_used

        recommendations.append({
            "title": movie_title,
            "overview": movie_overview,
            "genre": movie_genre,
            "similarity": round(similarities[idx], 2),
            "explanation": explanation.strip(),
            "tokens_used": tokens_used
        })

    return recommendations, total_tokens_used


# Streamlit UI
st.title("RAG-Based Movie Recommendation Engine")
st.write(
    "Get personalized movie recommendations based on your watch history or current mood, along with detailed explanations!"
)

# Option selection
option = st.selectbox(
    "Choose a recommendation type:",
    ("Suggest Movies Based on Current Mood", "Recommend Movies Based on Watch History")
)

# Mood-Based Recommendation Input
if option == "Suggest Movies Based on Current Mood":
    mood_query = st.text_input("Describe your mood or movie preferences.")

    if st.button("Get Mood-Based Recommendations") and mood_query:
        recommendations, total_tokens_used = recommend_with_explanations(mood_query=mood_query)
        for rec in recommendations:
            st.write(f"**Title**: {rec['title']}")
            st.write(f"**Overview**: {rec['overview']}")
            st.write(f"**Genre**: {rec['genre']}")
            st.write(f"**Explanation**: {rec['explanation']}")
            st.write(f"**Similarity Score**: {rec['similarity']}")
            st.write(f"**Tokens Used**: {rec['tokens_used']}")
            st.write("---")
        st.write(f"**Total Tokens Used for All Recommendations**: {total_tokens_used}")

# Watch History Recommendation Input
if option == "Recommend Movies Based on Watch History":
    st.write("List your top 5-6 favorite movies:")
    favorite_movies = [st.text_input(f"Movie {i + 1}", key=f"movie_{i + 1}") for i in range(5)]

    if st.button("Get Recommendations") and any(favorite_movies):
        recommendations, total_tokens_used = recommend_with_explanations(
            favorite_movies=[fm for fm in favorite_movies if fm])
        for rec in recommendations:
            st.write(f"**Title**: {rec['title']}")
            st.write(f"**Overview**: {rec['overview']}")
            st.write(f"**Genre**: {rec['genre']}")
            st.write(f"**Explanation**: {rec['explanation']}")
            st.write(f"**Similarity Score**: {rec['similarity']}")
            st.write(f"**Tokens Used**: {rec['tokens_used']}")
            st.write("---")
        st.write(f"**Total Tokens Used for All Recommendations**: {total_tokens_used}")
