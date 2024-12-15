import cv2
import streamlit as st
import numpy as np

face_cascade = cv2.CascadeClassifier('Face_recognition/haarcascade_frontalface_default.xml')

def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Create a placeholder for the video feed in Streamlit
    stframe = st.empty()

    while True:
        # Capture frames from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert frame from BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Exit the loop on pressing 'q' (but in Streamlit we can't use keypress in the same way)
        # Instead, we can stop after a certain number of frames or when the button is pressed

    # Release the webcam
    cap.release()

def app():
    st.title("Real-Time Face Detection")
    st.write("Click the button below to start detecting faces using your webcam.")

    # Button to trigger face detection
    if st.button("Start Detection"):
        detect_faces()

if __name__ == "__main__":
    app()
