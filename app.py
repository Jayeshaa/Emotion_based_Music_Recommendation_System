import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Reading the CSV file
df = pd.read_csv(r"C:\Users\Jayesha\OneDrive\Desktop\MusicProject\Music\muse_v3.csv")

df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']

df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

# Splitting the data into different emotion categories
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# Function to sample data based on emotion list
def fun(emotion_list):
    data = pd.DataFrame()
    if len(emotion_list) == 1:
        v = emotion_list[0]
        t = 30
        data = sample_data(v, t, data)
    elif len(emotion_list) == 2:
        times = [30, 20]
        for i in range(len(emotion_list)):
            v = emotion_list[i]
            t = times[i]
            data = sample_data(v, t, data)
    elif len(emotion_list) == 3:
        times = [55, 20, 15]
        for i in range(len(emotion_list)):
            v = emotion_list[i]
            t = times[i]
            data = sample_data(v, t, data)
    elif len(emotion_list) == 4:
        times = [30, 29, 18, 9]
        for i in range(len(emotion_list)):
            v = emotion_list[i]
            t = times[i]
            data = sample_data(v, t, data)
    else:
        times = [10, 7, 6, 5, 2]
        for i in range(len(emotion_list)):
            v = emotion_list[i]
            t = times[i]
            data = sample_data(v, t, data)
    return data

def sample_data(v, t, data):
    if v == 'Neutral':
        data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
    elif v == 'Angry':
        data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
    elif v == 'fear':
        data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
    elif v == 'happy':
        data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
    else:
        data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
    return data

def pre(l):
    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
    return ul

# Defining the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.load_weights(r'C:\Users\Jayesha\OneDrive\Desktop\MusicProject\Music\model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)

# Load Haarcascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

page_bg_img = '''
<style>
body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion based music recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the name of recommended song to reach website</b></h5>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

emotion_list = []
with col1:
    pass
with col2:
    if st.button('SCAN EMOTION(Click here)'):
        count = 0
        emotion_list.clear()

        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            count += 1

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))

                emotion_list.append(emotion_dict[max_index])

                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB")

            if count >= 20:
                break

        cap.release()

        emotion_list = pre(emotion_list)
        st.success(f"Emotions successfully detected, {emotion_list}")

with col3:
    pass

new_df = fun(emotion_list)
st.write("")

st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended song's with artist names</b></h5>", unsafe_allow_html=True)
st.write("---------------------------------------------------------------------------------------------------------------------")

try:
    for l, a, n, i in zip(new_df["link"], new_df['artist'], new_df['name'], range(30)):
        st.markdown(f"<h4 style='text-align: center;'><a href={l}>{i+1} - {n}</a></h4>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='text-align: center; color: grey;'><i>{a}</i></h5>", unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------------------------------------")
except Exception as e:
    st.write(f"An error occurred: {e}")
