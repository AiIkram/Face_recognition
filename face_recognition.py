import streamlit as st
import cv2
import numpy as np
import tempfile

def main():
    # Page Config
    st.set_page_config(page_title="Facial Detection")
    st.title("Facial Recognition Web App")
    st.caption("Powered by OpenCV, Streamlit")

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Video Capture
    cap = cv2.VideoCapture(0)  # Access webcam
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
        return

    frame_placeholder = st.empty()  # Placeholder for video frames
    stop_button_pressed = st.button("Stop")  # Stop button

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_coordinates = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw rectangles around detected faces
        for (fx, fy, fw, fh) in face_coordinates:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        # Display frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")

        # Stop if the stop button is pressed
        if stop_button_pressed:
            st.write("Stopping the video...")
            break

    cap.release()  # Release the video capture object

if __name__ == "__main__":
    main()
