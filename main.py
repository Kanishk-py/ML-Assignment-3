import numpy as np
import cv2
from keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

emotions = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

model = load_model('fer_model.h5')
model.load_weights('./face_emotion_model.h5', by_name=True)
def predict_emotion(image):
	return emotions[np.argmax(model.predict(np.asarray(image)))]


scale = 1.5
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

            cv2.rectangle(frame, (x_new, y_new), (x_new+w_new, y_new+h_new), (255, 0, 0), 2)

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
    activiteis = ["Home", "About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)

    # Home
    if choice == "Home":
        st.title("Face Emotion Recognition Application.")
        html_temp_home1 = """
                        <div style="background-color:#FC4C02;padding:0.5px">
                            <h4 style="color:white;text-align:center;">
                                Face Emotion Recognition Application Using OpenCV and Tensorflow
                            </h4>
                        </div>
                        </br>
                        """

        st.markdown(html_temp_home1, unsafe_allow_html=True)
                
        st.header("Webcam Live Feed")
        st.write("1. Click Start to open your camera and give permission for prediction")
        st.write("2. This will predict your emotion.")
        st.write("3. When you done, click stop to end.")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    # About.
    elif choice == "About":
        st.title("About this app")
        st.write("""
                This app predicts facial emotion using a Convolutional neural network.
                Which is built using Keras and Tensorflow libraries.
                Face detection is achived through openCV.
                """)
        st.subheader("About Developers")
        about_html =  """
                This app is developed:<br/>
                - Kanishk Singhal (kanishk.singhal@iitgn.ac.in)
                - Lipika Rajpal (lipika.rajpal@iitgn.ac.in)
                """
        st.markdown(about_html, unsafe_allow_html=True)
    else:
        pass


if __name__ == "__main__":
    main()