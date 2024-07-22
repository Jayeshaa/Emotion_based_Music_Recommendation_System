# Emotion_based_Music_Recommendation_System

The Emotion-Based Music Recommendation System is an innovative application designed to personalize music recommendations based on the user's current emotional state. This system combines the power of deep learning for emotion detection and music recommendation, utilizing data fetched from the Spotify API, and delivers the recommendations through an interactive Streamlit web application.

# Key Features

Emotion Detection

Convolutional Neural Network (CNN): The CNN model processes images to detect and classify the user's emotional state. This model is trained to recognize various emotions from facial expressions, ensuring accurate emotion detection.

Music Recommendation

Long Short-Term Memory (LSTM): The LSTM model is used to recommend songs based on the detected emotions. It takes into account the emotional context and suggests music that aligns with the user's current mood.

Spotify API Integration

Spotify API: The system fetches song data, including tracks, artists, and playlists, from the Spotify API. This integration ensures that the recommendations are up-to-date and includes a wide variety of music genres.

User-Friendly Interface

Streamlit: The application is deployed using Streamlit, providing an intuitive and interactive web interface. Users can easily upload images, view their detected emotions, and receive personalized music recommendations.


# How It Works

Emotion Detection:

Users upload an image of their face through the Streamlit interface.
The image is processed using a CNN model, which detects and classifies the user's emotion (e.g., happy, sad, angry).

Music Recommendation:

Based on the detected emotion, the LSTM model recommends a list of songs that match the user's mood.
The system fetches the recommended songs from the Spotify API, ensuring a diverse and relevant selection.

Interactive Interface:

Users can interact with the web application to see their detected emotions and the corresponding music recommendations.
The app provides links to play the recommended songs directly on Spotify.

# Benefits

Personalization: Tailors music recommendations to the user's current emotional state, enhancing their listening experience.

Accuracy: Utilizes advanced deep learning models to ensure accurate emotion detection and relevant music suggestions.

Convenience: Provides an easy-to-use web interface, making it accessible for users to receive personalized music recommendations effortlessly.

# Applications

The Emotion-Based Music Recommendation System is ideal for music streaming services, wellness apps, and any application where personalized music recommendations can enhance user engagement and satisfaction. It offers a unique way to connect users with music that resonates with their emotions, providing a more meaningful and enjoyable listening experience.

# Conclusion

By combining deep learning techniques with real-time data from the Spotify API, the Emotion-Based Music Recommendation System offers a cutting-edge solution for personalized music recommendations. Its user-friendly interface and accurate emotion detection make it a valuable tool for anyone looking to enhance their music listening experience based on their emotional state.


# Screenshots



![pic1_emrs](https://github.com/user-attachments/assets/da6da556-e85b-4b35-997f-3fbf03c4f689)

![pic2_emrs](https://github.com/user-attachments/assets/285c94d0-f6eb-4a75-a6a5-f870f9f23bf1)

![pic3_emrs](https://github.com/user-attachments/assets/f9916b6a-2339-43ab-a5de-597844078077)


