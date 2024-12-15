

from cv2 import cv2
import streamlit as st


face_cascade = cv2.CascadeClassifier('https://raw.githubusercontent.com/opencv/opencv/refs/heads/master/data/haarcascades/haarcascade_frontalface_default.xml')

def detect_faces():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frames from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Face Detection', frame)

        # Exit the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def app():
    st.title("Real-Time Face Detection")
    st.write("Click the button below to start detecting faces using your webcam.")

    # Button to trigger face detection
    if st.button("Start Detection"):
        detect_faces()

if __name__ == "__main__":
    app()



