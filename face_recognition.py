import cv2
import streamlit as st
import tempfile
import threading
import time

def detect_faces(video_path, cascade_path, frame_callback, stop_event):
    """Run face detection on frames and send results to the callback."""
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Send the processed frame to the callback
        frame_callback(frame)

        # Add a small delay to simulate processing time
        time.sleep(0.03)  # 30 FPS

    cap.release()

def main():
    st.set_page_config(page_title="Facial Detection", layout="centered")
    st.title("Fast Facial Detection with Display")
    st.caption("Powered by OpenCV and Streamlit")

    cascade_path = "./haarcascade_frontalface_default.xml"

    # Upload the video file
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        st.video(video_path)  # Quickly play the original video

        # Display the processed video frames
        st.write("Processed Frames (with Face Detection):")
        frame_placeholder = st.empty()

        # Stop processing button
        stop_button = st.button("Stop Detection")
        stop_event = threading.Event()

        def frame_callback(processed_frame):
            """Update the Streamlit placeholder with the processed frame."""
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        if not stop_button:
            # Start the face detection thread
            threading.Thread(
                target=detect_faces, args=(video_path, cascade_path, frame_callback, stop_event), daemon=True
            ).start()
        else:
            stop_event.set()
            st.write("Face detection stopped.")

if __name__ == "__main__":
    main()
