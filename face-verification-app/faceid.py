from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
import numpy as np
import os
from layers import L1dist

class CampApp(App):
    def build(self):

        # app layout
        self.cap=cv2.VideoCapture(0)
        self.model=tf.keras.models.load_model('your model here',custom_objects={'L1dist':L1dist}) ###paste your model
        self.webcam=Image(size_hint=(1,.8))
        self.verification_label=Label(text='unintiated',size_hint=(1,.1))
        self.button=Button(text="verify",on_press=self.verify,size_hint=(1,.1))
        
        layout=BoxLayout(orientation= 'vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        # camera stuff

        Clock.schedule_interval(self.update,1/33)
        return layout

    def update(self,*args):
        ret , frame= self.cap.read()
        frame = frame[100:350, 200:450,:]

        # convert raw img to kivy texture so we can use it
        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
        self.webcam.texture=img_texture

    #convert filepath to jpg format
    def preprocessing(self,file_path):
        #get binary image 
        bin_img=tf.io.read_file(file_path)
        #convert to jpeg 
        img=tf.io.decode_jpeg(bin_img)
        img=tf.image.resize(img,(100,100))
        #scaling
        img=img/255.0
        return img
    
    # functon to verify the image
    def verify(self,*args):
        # save the img
        ret,frame=self.cap.read()
        frame = frame[100:350, 200:450]
        cv2.imwrite(os.path.join('app_live', 'input_img', 'input_image.jpg'),frame)

        results = []
        detection_thresh=0.5
        validation_thresh=0.6
        validation_dir = os.path.join('app_live', 'validation_img')
        validation_imgs = os.listdir(validation_dir)
        for image in validation_imgs:
            input_img = self.preprocessing(
                os.path.join('app_live', 'input_img', 'input_image.jpg')
            )
            verification_img = self.preprocessing(
                os.path.join(validation_dir, image)
            )
            input_img = input_img[None, ...]
            verification_img = verification_img[None, ...]
            score = self.model(
                [input_img, verification_img],
                training=False
            ).numpy()[0][0]
            results.append(score)
        results = np.array(results)
        detection = np.sum(results > detection_thresh)
        verification = (detection / len(results)) >= validation_thresh

        # update verification label
        self.verification_label.text="Verified" if verification == True else "Unverified"
        Logger.info(results)
        Logger.info(np.sum(results > 0.3))
        Logger.info(np.sum(results > 0.4))
        Logger.info(np.sum(results > 0.5))
        Logger.info(np.sum(results > 0.6))
        Logger.info(np.sum(results > 0.7))
        Logger.info(np.sum(results > 0.8))
        return results, verification


CampApp().run()