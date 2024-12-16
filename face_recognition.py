import cv2
import streamlit as st
import tempfile

def main():
    # Page Configuration for Streamlit
    st.set_page_config(page_title="Facial Detection", layout="centered")
    st.title("Facial Detection Web App")
    st.caption("Powered by OpenCV and Streamlit")

    # Load Haar Cascade for face detection
    cascade_path = "./haarcascade_frontalface_default.xml"  # Adjust path if necessary
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            st.error("Error loading Haar Cascade file. Ensure the XML file is in the correct directory.")
            return
    except Exception as e:
        st.error(f"Error loading Haar Cascade: {e}")
        return

    # File uploader widget to allow users to upload a video
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if video_file is not None:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(video_file.read())
            video_path = temp_file.name

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Unable to open the video file. Please check the file format or try again.")
            return

        st.write("Processing the video. Please wait...")
        frame_placeholder = st.empty()  # Placeholder for displaying frames

        # Process the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to speed up processing
            frame = cv2.resize(frame, (640, 360))

            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            face_coordinates = face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # Draw rectangles around detected faces
            for (fx, fy, fw, fh) in face_coordinates:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        cap.release()
        st.success("Video processing complete!")

if __name__ == "__main__":
    main()
