import cv2
import streamlit as st
import tempfile
import numpy as np

def main():
    # Page Configuration for Streamlit
    st.set_page_config(page_title="Facial Recognition", layout="centered")
    st.title("Facial Recognition Web App")
    st.set_page_config(page_title="Facial Detection", layout="centered")
    st.title("Facial Detection Web App")
    st.caption("Powered by OpenCV and Streamlit")

    # Load Haar Cascade for face detection

                st.warning("Failed to grab a frame or video finished. Exiting...")
                break

            # Convert the frame to grayscale
            # Convert the frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            face_coordinates = face_cascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            # Draw rectangles around detected faces
            # Draw rectangles around detected faces (Face Detection)
            for (fx, fy, fw, fh) in face_coordinates:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

            # Convert the BGR frame to RGB (Streamlit expects RGB format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Display the frame in Streamlit
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Check if the stop button is pressed
            if stop_button:
                st.write("Stopping video stream...")
                break

        # Release resources
        cap.release()
        st.write("Video capture stopped.")

if __name__ == "__main__":
    main()
