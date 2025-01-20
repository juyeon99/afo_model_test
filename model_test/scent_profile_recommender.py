import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def update_user_profile(user_id, scent_feedback, profiles_df):
    """
    Update user profile based on feedback.

    Parameters:
    - user_id (int): ID of the user.
    - scent_feedback (dict): Feedback with scent categories and scores (e.g., {'citrus': 1, 'floral': -1}).
    - profiles_df (pd.DataFrame): DataFrame containing user profiles with scent preferences.

    Returns:
    - Updated profiles DataFrame.
    """
    if user_id not in profiles_df.index:
        profiles_df.loc[user_id] = 0  # Initialize profile if user not in DataFrame

    for scent, feedback in scent_feedback.items():
        profiles_df.loc[user_id, scent] += feedback

    # Normalize the user's profile (optional, keeps values between 0 and 1)
    profiles_df.loc[user_id] = profiles_df.loc[user_id].clip(lower=0)

    return profiles_df

def recommend_scents(user_id, profiles_df, scent_db, top_n=5):
    """
    Recommend scents based on user profile.

    Parameters:
    - user_id (int): ID of the user.
    - profiles_df (pd.DataFrame): DataFrame containing user profiles.
    - scent_db (pd.DataFrame): DataFrame with scent metadata (e.g., categories).
    - top_n (int): Number of recommendations to return.

    Returns:
    - List of recommended scents.
    """
    if user_id not in profiles_df.index:
        return []  # No profile data for this user

    user_profile = profiles_df.loc[user_id].values.reshape(1, -1)
    scent_vectors = scent_db.iloc[:, 1:].values  # Assuming first column is scent name

    # Calculate similarity between user profile and scents
    similarities = cosine_similarity(user_profile, scent_vectors).flatten()
    scent_db["similarity"] = similarities

    # Sort by similarity and return top_n scents
    recommendations = scent_db.sort_values("similarity", ascending=False).head(top_n)
    return recommendations["scent_name"].tolist()

# Example setup
user_profiles = pd.DataFrame(columns=["citrus", "floral", "woody", "spicy"])
scent_database = pd.DataFrame({
    "scent_name": ["Fresh Citrus", "Rose Garden", "Woody Musk", "Spicy Vanilla"],
    "citrus": [1, 0, 0, 0],
    "floral": [0, 1, 0, 0],
    "woody": [0, 0, 1, 0],
    "spicy": [0, 0, 0, 1],
})

# Example feedback and usage
user_id = 123
feedback = {"citrus": 1, "floral": -1}  # User likes citrus, dislikes floral
user_profiles = update_user_profile(user_id, feedback, user_profiles)
recommendations = recommend_scents(user_id, user_profiles, scent_database)

print("User Profile:")
print(user_profiles)
print("\nRecommended Scents:")
print(recommendations)
