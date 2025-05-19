import pandas as pd
import random
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
# Step 1: Load the Spotify dataset from a CSV file
df = pd.read_csv('dataset (1).csv')

# Step 2: Data Preprocessing
# We will use the columns: danceability, energy, popularity, and explicit for the model.
# Encode the explicit column (True/False) as 1/0
df['explicit'] = df['explicit'].astype(int)

# Step 3: Feature Selection and Scaling
# We will scale the features that are numeric (danceability, energy, and popularity)
features = ['danceability', 'energy', 'popularity', 'explicit']
X = df[features]

# Standardizing the features (scaling them to have mean 0 and variance 1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Training a K-Nearest Neighbors model
# Using KNN to find similar songs based on the selected features
knn_model = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean')
knn_model.fit(X_scaled)

# Step 5: Recommend songs based on user preferences using KNN
def recommend_based_on_user_preferences(user_danceability, user_energy, user_popularity, user_explicit):
    # User's preference vector
    user_preferences = [[user_danceability, user_energy, user_popularity, user_explicit]]
    
    # Scale the user's preferences using the same scaler
    user_preferences_scaled = scaler.transform(user_preferences)
    
    # Find the nearest neighbors (songs similar to user's preferences)
    distances, indices = knn_model.kneighbors(user_preferences_scaled)
    
    # Get the recommended songs based on nearest neighbors
    recommended_songs = df.iloc[indices[0]]
    
    # Return the recommended songs
    return recommended_songs[['track_name', 'artists', 'album_name', 'popularity', 'danceability', 'energy']]

# Step 6: Get current season (used for seasonal playlist, no impact on machine learning)
def get_season():
    current_month = datetime.now().month
    if 3 <= current_month <= 5:
        return 'Spring'
    elif 6 <= current_month <= 8:
        return 'Summer'
    elif 9 <= current_month <= 11:
        return 'Autumn'
    else:
        return 'Winter'

# Step 7: Generate a seasonal playlist (Random sampling for simplicity)
def seasonal_playlist(df):
    current_season = get_season()
    print(f"Current Season: {current_season}")
    seasonal_songs = df.sample(n=10)  # Random 10 songs for the playlist (no actual season data in the dataset)
    
    seasonal_playlist = seasonal_songs[['track_name', 'artists', 'album_name', 'popularity', 'danceability', 'energy']]
    
    return seasonal_playlist

# Step 8: Generate a daily playlist based on user preferences
def daily_playlist(df, user_danceability, user_energy):
    preferred_songs = df[(df['danceability'] >= user_danceability) & (df['energy'] >= user_energy)]
    daily_playlist = preferred_songs[['track_name', 'artists']].sample(n=7, replace=False).reset_index(drop=True)
    return daily_playlist

# Step 9: User Interaction - Simulating user preferences and making recommendations
def get_user_recommendations(df):
    print("Enter your preferred minimum danceability (0-1): ")
    user_danceability = float(input().strip())
    
    print("Enter your preferred minimum energy (0-1): ")
    user_energy = float(input().strip())
    
    print("Enter your minimum popularity preference (0-100): ")
    user_popularity = float(input().strip())
    
    print("Do you prefer explicit content? (Yes/No): ")
    user_explicit = 1 if input().strip().lower() == 'yes' else 0
    
    # Get recommendations using the KNN model
    recommendations = recommend_based_on_user_preferences(user_danceability, user_energy, user_popularity, user_explicit)
    
    return recommendations

# Step 10: Main Function to Run the Recommendation System
def music_recommendation_system(df):
    seasonal_songs = seasonal_playlist(df)
    print("\nSeasonal Music Playlist:")
    print(seasonal_songs)
    
    print("\nGet Recommendations Based on Your Preferences:")
    recommendations = get_user_recommendations(df)
    print("\nRecommended Songs based on Your Preferences:")
    print(recommendations)
    
    print("\nGenerating  Your Daily Playlist:")
    daily_songs = daily_playlist(df, recommendations.iloc[0]['danceability'], recommendations.iloc[0]['energy'])
    print(daily_songs)
    
    return df

# Running the music recommendation system
df = music_recommendation_system(df)


