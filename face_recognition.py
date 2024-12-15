import streamlit as st
import cv2
import numpy as np

def main():
    # Page Configuration
    st.set_page_config(page_title="Facial Detection", layout="centered")
    st.title("Facial Recognition Web App")
    st.caption("Powered by OpenCV and Streamlit")

    # Load Haar Cascade for face detection
    cascade_path = "./haarcascade_frontalface_default.xml"
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            st.error("Error loading Haar Cascade file. Ensure the XML file is in the correct directory.")
            return
    except Exception as e:
        st.error(f"Error loading Haar Cascade: {e}")
        return

    # Video Capture Setup
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)  # Try changing to 0, 1, 2 if needed
    if not cap.isOpened():
        st.error("Unable to access the webcam. Ensure your camera is connected and permissions are granted.")
        print("Failed to open webcam.")  # Print debug message
        return
    else:
        print("Webcam opened successfully.")  # Debug message

    # Check camera resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera resolution: {width} x {height}")

    st.write("Press the **Stop** button to end the video.")
    
    frame_placeholder = st.empty()  # Placeholder for video frames
    stop_button = st.button("Stop")  # Stop button

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab a frame. Exiting...")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        face_coordinates = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Draw rectangles around detected faces
        for (fx, fy, fw, fh) in face_coordinates:
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

        # Display the frame in Streamlit
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Check if the stop button is pressed
        if stop_button:
            st.write("Stopping video stream...")
            break

    # Release resources
    cap.release()
    st.write("Video capture stopped.")

if __name__ == "__main__":
    main()
