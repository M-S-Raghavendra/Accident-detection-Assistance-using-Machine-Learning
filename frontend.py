import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from twilio.rest import Client
import keys
from PIL import Image

def callEmergencyServices():
       client = Client(keys.acount_sid,keys.auth_token)

       call = client.calls.create(
            url='http://demo.twilio.com/docs/voice.xml',
            to=keys.my_number,
            from_= keys.twilio_number
       )

loaded_model = tf.keras.models.load_model("mymodel.h5")

def predict_accident(filename):
    img_width, img_height = 224, 224
    img = image.load_img((filename), target_size = (img_width, img_height, 3))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    
    prediction = loaded_model.predict(img)
    predicted_class = np.argmax(prediction)
    #if(predicted_class==1):
        #print("No accident")
    if(predicted_class==0):
        if(prediction[0][predicted_class]>0.8): 
            acc_image =  Image.open(filename)
            st.image(acc_image, caption="Accident detected frame")
            return "Accident Predicted with accuracy "+ str(prediction[0][predicted_class])
    return ""

def frameCapture(path):
    
    success=1
    count = 0
    acc = False
    video = cv2.VideoCapture(path)

    while(success):
        count+=1
        success, image = video.read()
        cv2.imwrite("../data/frames/acc_4_"+str(count)+".jpg",image)
        ret = predict_accident("../data/frames/acc_4_"+str(count)+".jpg")
        if(ret!=""):
          acc=True
          st.write(ret)
          st.write("Calling emergency services")
          callEmergencyServices()
          break
    if(acc==False):
       st.write("No accident")

st.title("Accident detection using Transfer learning")

uploaded_video = st.file_uploader("Choose a video", type=".mp4")

if(uploaded_video):
     vid = uploaded_video.name
     with open(vid, mode='wb') as f:
        f.write(uploaded_video.read())
     st.write("Video uploaded successfully")
     st.write("Predicting accident......")
     frameCapture(vid)