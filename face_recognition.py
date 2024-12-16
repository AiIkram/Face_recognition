import cv2
import streamlit as st
import tempfile
import threading

def detect_faces(video_path, cascade_path, frame_callback):
    """Run face detection on frames and send results to the callback."""
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Send the processed frame to the callback
        frame_callback(frame)

    cap.release()

def main():
    st.set_page_config(page_title="Facial Detection", layout="centered")
    st.title("Fast Facial Detection Web App")
    st.caption("Powered by OpenCV and Streamlit")

    cascade_path = "./haarcascade_frontalface_default.xml"

    # Upload the video file
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        st.video(video_path)  # Quickly play the video while processing frames

        # Display the processed video in a separate section
        st.write("Processed Frames (with Face Detection):")
        frame_placeholder = st.empty()

        # Process video frames asynchronously
        def frame_callback(processed_frame):
            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        threading.Thread(
            target=detect_faces, args=(video_path, cascade_path, frame_callback), daemon=True
        ).start()

if __name__ == "__main__":
    main()
