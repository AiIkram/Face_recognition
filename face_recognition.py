import streamlit as st
import streamlit.components.v1 as components
import cv2
import logging as log
import datetime as dt
from time import sleep
from deepface import DeepFace


cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

st.title("HP's Face Detection Project !")

st.write("The app uses Haar Cascade Classifier. It is a machine learning based approach where a "
         "cascade function is trained from a lot of positive and negative images which is then used to "
         "detect objects in other images.")

st.write("The app also uses DeepFace to detect your current emotion, gender and race. This is a very simple opencv application")

# Building a sidebar
st.sidebar.subheader("Details of the person")
t1 = st.sidebar.text_input("Name of the Person ")
s1 = st.sidebar.slider("Age of the person ")
r1 = st.sidebar.selectbox("Race:",['Asian','Indian','Black','White','Middle Eastern','Latino Hispanic'] )


st.subheader('Details of the User:')
st.write("Name: ",t1)
st.write("Age: ", s1) 
st.write("Race: ", r1)# taking data from the sidebar




if st.button("Can I detect your face ?"):
    st.write('Press "q" to close the Camera Window')
    # Selection box
    # first argument takes the titleof the selectionbox second argument takes options
    How_is_Project = st.selectbox("How did the Face Detection Work : ",['Choose an Option :','Useless', 'So so', 'Good', 'Perfect'])
    st.write("Thank you")
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera.')    
            sleep(5)
            pass

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        result = DeepFace.analyze(frame, actions =['emotion','gender','age','race'], enforce_detection= False)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame,result['dominant_emotion'],(50,50),font,2,(0,0,255),2,cv2.LINE_4)
        cv2.putText(frame,str(result['age']),(50,100),font,2,(0,0,255),2,cv2.LINE_4)
        cv2.putText(frame,result['gender'],(50,250),font,2,(0,0,255),2,cv2.LINE_4)
        cv2.putText(frame,result['dominant_race'],(50,400),font,2,(0,0,255),2,cv2.LINE_4)



        if anterior != len(faces):
            anterior = len(faces)
            log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
