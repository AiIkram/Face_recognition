import cv2
import streamlit as st
import tempfile
import threading
import time
import os

def detect_faces_and_stream(video_path, cascade_path, stop_event):
    """Run face detection on frames and stream video output."""
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(video_path)

    # Check if video was opened successfully
    if not cap.isOpened():
        raise Exception("Error: Unable to open the video file.")

    # Temporary file to store processed video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_output.name
    temp_output.close()  # Close the temp file so OpenCV can write to it

    # Video Writer for processed frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if FPS is 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=3)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write processed frame to output video
        out.write(frame)

    cap.release()
    out.release()
    return output_path

def main():
    st.set_page_config(page_title="Facial Detection Video App", layout="centered")
    st.title("Fast Facial Detection with Video Stream")
    st.caption("Powered by OpenCV and Streamlit")

    cascade_path = "./haarcascade_frontalface_default.xml"

    # Upload the video file
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        # Display the original video
        st.video(video_path)

        # Stop detection button
        stop_button = st.button("Stop Detection")
        stop_event = threading.Event()

        if stop_button:
            stop_event.set()
            st.write("Face detection stopped.")
        else:
            # Start face detection and video streaming
            with st.spinner("Processing video for face detection..."):
                try:
                    processed_video_path = detect_faces_and_stream(video_path, cascade_path, stop_event)
                    st.success("Face detection complete! Processed video displayed.")
                    st.video(processed_video_path)
                except Exception as e:
                    st.error(f"Error processing video: {e}")

if __name__ == "__main__":
    main()
