import cv2
import streamlit as st

def main():
    # Page Configuration for Streamlit
    st.set_page_config(page_title="Facial Recognition", layout="centered")
    st.title("Facial Recognition Web App")
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

    # Video File Path
    video_path = "video.mp4"  # Specify the path to your video file

    # Load the video file
    cap = cv2.VideoCapture(video_path)  # Use the video file path instead of a camera index
    if not cap.isOpened():
        st.error("Unable to open the video file. Please check the file path.")
        return

    st.write("Press the **Stop** button to end the video.")

    frame_placeholder = st.empty()  # Placeholder for video frames
    stop_button = st.button("Stop")  # Stop button

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab a frame or video finished. Exiting...")
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
