import cv2
import streamlit as st
import tempfile
import time

# Initialize the face cascade
cascade_path = "./haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces_from_video(video_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open the video file.")
        return

    st.write("Press **Stop** to exit.")
    frame_placeholder = st.empty()
    stop_button = st.button("Stop")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert the frame to RGB (Streamlit uses RGB, not BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Check if the stop button is pressed
        if stop_button:
            st.write("Stopping...")
            break

        time.sleep(0.03)  # Simulate ~30 FPS

    cap.release()
    st.write("Video processing stopped.")

def main():
    st.set_page_config(page_title="Facial Detection", layout="centered")
    st.title("Facial Detection Web App")
    st.caption("Powered by OpenCV and Streamlit")

    # Upload video
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
        detect_faces_from_video(video_file)

if __name__ == "__main__":
    main()
