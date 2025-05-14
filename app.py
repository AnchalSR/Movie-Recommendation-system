import streamlit as st
import pandas as pd
import numpy as np
import pickle
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Add CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 30px;
        font-weight: bold;
    }
    .movie-poster {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s;
    }
    .movie-poster:hover {
        transform: scale(1.05);
    }
    .movie-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 10px;
        text-align: center;
    }
    .recommendation-section {
        margin-top: 40px;
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(70, 70, 70, 0.1);
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .stSelectbox>div>div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
    }
    .search-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    footer {
        text-align: center;
        margin-top: 50px;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract data from nested JSON in string format
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Function to convert 3 cast members
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

# Function to get director name from crew
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Function to get movie poster from TMDB API
@st.cache_data(ttl=3600*24)  # Cache poster URLs for 24 hours
def fetch_poster(movie_id):
    max_retries = 3
    retries = 0
    
    while retries < max_retries:
        try:
            url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                print(f"Error fetching movie {movie_id}: Status code {response.status_code}")
                return "https://via.placeholder.com/500x750?text=No+Poster+Available"
                
            data = response.json()
            if 'poster_path' in data and data['poster_path']:
                poster_path = data['poster_path']
                full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
                return full_path
            else:
                return "https://via.placeholder.com/500x750?text=No+Poster+Available"
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries >= max_retries:
                print(f"Failed to fetch poster for movie ID {movie_id} after {max_retries} attempts: {e}")
                return "https://via.placeholder.com/500x750?text=No+Poster+Available"
            print(f"Retry {retries}/{max_retries} for movie ID {movie_id}")
        except Exception as e:
            print(f"Unexpected error fetching poster for movie ID {movie_id}: {e}")
            return "https://via.placeholder.com/500x750?text=No+Poster+Available"
    
    return "https://via.placeholder.com/500x750?text=No+Poster+Available"

# Load data function
@st.cache_data
def load_data():
    # Read movies and credits data
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    
    # Merge data
    movies = movies.merge(credits, on='title')
    
    # Select important columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Drop rows with missing data
    movies.dropna(inplace=True)
    
    # Convert string representations to lists
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Create tags by combining features
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
    
    # Combine all features into tags
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create a new dataframe with only needed columns - use copy() to avoid SettingWithCopyWarning
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    
    # Convert tags list to string
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    
    # Convert tags to lowercase
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    # Create count vector features
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    # Calculate similarity scores
    similarity = cosine_similarity(vectors)
    
    return new_df, similarity

# Movie recommendation function
def recommend(movie, data, similarity_matrix):
    if movie not in data['title'].values:
        return []
    
    index = data[data['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
    
    recommended_movies = []
    for i in distances[1:6]:  # Get top 5 similar movies
        movie_id = data.iloc[i[0]].movie_id
        title = data.iloc[i[0]].title
        poster = fetch_poster(movie_id)
        recommended_movies.append({
            'title': title,
            'poster': poster,
            'movie_id': movie_id
        })
    
    return recommended_movies

# Function to save processed data for faster loading
def save_data(dataframe, similarity_matrix):
    try:
        # Save dataframe
        dataframe.to_pickle('movie_data.pkl')
        
        # Save similarity matrix
        with open('similarity.pkl', 'wb') as f:
            pickle.dump(similarity_matrix, f)
            
        st.success("Data saved successfully!")
    except Exception as e:
        st.error(f"Error saving data: {e}")

# Function to load saved data
@st.cache_resource
def load_saved_data():
    try:
        # Check if saved files exist
        if os.path.exists('movie_data.pkl') and os.path.exists('similarity.pkl'):
            # Load dataframe
            data = pd.read_pickle('movie_data.pkl')
            
            # Load similarity matrix
            with open('similarity.pkl', 'rb') as f:
                sim_matrix = pickle.load(f)
                
            return data, sim_matrix, True
        else:
            return None, None, False
    except Exception as e:
        st.error(f"Error loading saved data: {e}")
        return None, None, False

# Main app
st.markdown("<h1 class='main-header'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

# Load data
with st.spinner('Loading movie data... This might take a moment.'):
    # First try to load saved data
    movie_data, similarity, data_loaded = load_saved_data()
    
    # If saved data doesn't exist, process it
    if not data_loaded:
        movie_data, similarity = load_data()
        
        # Add a button to save the processed data
        if st.button("Save processed data for faster loading"):
            save_data(movie_data, similarity)

# Get the list of all movies
movie_list = movie_data['title'].tolist()
movie_list.sort()  # Sort alphabetically for easier selection

# Create a centered container for the search box
st.markdown("<div class='search-container'>", unsafe_allow_html=True)

# Add a subtitle
st.markdown("<h3 style='text-align: center; margin-bottom: 20px;'>Find movie recommendations based on your favorite film</h3>", unsafe_allow_html=True)

# Create the search box
selected_movie = st.selectbox(
    "Type or select a movie you like",
    movie_list
)

# Add a button to get recommendations
if st.button('Show Recommendations', key='recommend_button'):
    # Close the container div
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.spinner('Finding movies you might like...'):
        recommendations = recommend(selected_movie, movie_data, similarity)
        
        if recommendations:
            st.markdown("<div class='recommendation-section'>", unsafe_allow_html=True)
            st.subheader(f"Top 5 movies similar to '{selected_movie}'")
            
            # Display selected movie poster
            selected_movie_id = movie_data[movie_data['title'] == selected_movie]['movie_id'].values[0]
            selected_movie_poster = fetch_poster(selected_movie_id)
            
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown(f"""
                <div style="text-align: center;">
                    <img src="{selected_movie_poster}" width="200" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
                    <p style="margin-top: 8px; font-weight: bold;">Your selected movie</p>
                </div>
                """, unsafe_allow_html=True)
            
            with cols[1]:
                st.write("### Based on this movie, you might also like:")
                # Create a container for recommendations to apply custom styling
                rec_container = st.container()
                rec_cols = rec_container.columns(5)
                
                for i, movie in enumerate(recommendations):
                    with rec_cols[i]:
                        # Apply custom CSS class to each image for better styling
                        st.markdown(f"""
                        <div style="text-align: center;">
                            <img src="{movie['poster']}" width="150" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);">
                            <p style="margin-top: 8px; font-weight: bold;">{movie['title']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Sorry, we couldn't find recommendations for this movie. Please try another one.")
else:
    # Close the container div if button isn't clicked
    st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown("""
---
Created with ❤️ using Streamlit | Data source: TMDB
""")
