import cv2
import streamlit as st
import tempfile
import time

def main():
    st.set_page_config(page_title="Facial Detection", layout="centered")
    st.title("Facial Detection Web App")
    st.caption("Powered by OpenCV and Streamlit")

    cascade_path = "./haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
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

        frame_skip = 2
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (frame_count % frame_skip != 0):
                frame_count += 1
                continue

            frame = cv2.resize(frame, (640, 360))  # Resize for faster processing
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.2, 3)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            if stop_button:
                st.write("Stopping...")
                break

            time.sleep(0.03)  # Simulate ~30 FPS

        cap.release()
        st.write("Video processing stopped.")

if __name__ == "__main__":
    main()
