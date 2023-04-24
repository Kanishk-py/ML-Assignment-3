# import the rquired libraries.
import numpy as np
import cv2
from keras.models import load_model
import streamlit as st
from tensorflow import keras
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# Define the emotions.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Load model.
model = load_model('fer_model.h5')

# load weights into new model
model.load_weights('./face_emotion_model.h5', by_name=True)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
def predict_emotion(image):
	return emotions[np.argmax(model.predict(np.asarray(image)))]
scale = 1.5
# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            x_new = int(x - (scale - 1) / 2 * w)
            y_new = int(y - (scale - 1) / 2 * h)
            w_new = int(scale * w)
            h_new = int(scale * h)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x_new, y_new), (x_new+w_new, y_new+h_new), (255, 0, 0), 2)

            # Crop the face from the frame
            face_image = gray[y_new:y_new+h_new, x_new:x_new+w_new] 

            try:
                face_image = cv2.resize(face_image, (48, 48))
                emo = predict_emotion(face_image.reshape(1, 48, 48, 1))
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
                cv2.putText(frame, emo, (int(x_new+w_new/3), y_new+h_new+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            except Exception as e:
                # print(e)
                continue
            cv2.imshow('Cropped Face', face_image)
        return frame

def main():
    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application 😠🤮😨😀😐😔😮")
    activiteis = ["Home", "Live Face Emotion Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    st.sidebar.markdown(
        """ Developed by Kanishk Singhal and Lipika Rajpal
            [LinkedIn](https://www.linkedin.com/in/anish-johnson-594110208/)""")

    # Homepage.
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#FC4C02;padding:0.5px">
                             <h4 style="color:white;text-align:center;">
                            Start Your Real Time Face Emotion Detection.
                             </h4>
                             </div>
                             </br>"""

        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
        * An average human spends about 10 to 15hrs a day staring at a computer screen, during which our facial expressions keep on changing. 
        * Sometimes we laugh, sometimes we cry, sometimes we get angry, and sometimes get scared by our face when the camera turns on accidentally.
        * But ever wondered; whether the computer that we give all this attention to is even capable of recognizing these emotions?
        
        Let's find out...
        1. Click the dropdown list in the top left corner and select Live Face Emotion Detection.
        2. This takes you to a page which will tell if it recognizes your emotions.
                 """)

    # Live Face Emotion Detection.
    elif choice == "Live Face Emotion Detection":
        st.header("Webcam Live Feed")
        st.subheader('''
        Welcome to the other side of the SCREEN!!!
        * Get ready with all the emotions you can express. 
        ''')
        st.write("1. Click Start to open your camera and give permission for prediction")
        st.write("2. This will predict your emotion.")
        st.write("3. When you done, click stop to end.")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    # About.
    elif choice == "About":
        st.subheader("About this app")
        html_temp_about1= """<div style="background-color:#36454F;padding:30px">
                                    <h4 style="color:white;">
                                     This app predicts facial emotion using a Convolutional neural network.
                                     Which is built using Keras and Tensorflow libraries.
                                     Face detection is achived through openCV.
                                    </h4>
                                    </div>
                                    </br>
                                    """
        st.markdown(html_temp_about1, unsafe_allow_html=True)


    else:
        pass


if __name__ == "__main__":
    main()