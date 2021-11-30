# Import kivy dependencies first
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Import kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

# Import other kivy stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# Import other dependencies
import cv2
import tensorflow as tf
import os
import numpy as np
import mediapipe as mp
import pandas as pd

# Build app and layout

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class AmericanSignLanguageRecognitionApp(App):

    def build(self):
        # Main layout components
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(
            text="Prediction Using EfficientNet Model", on_press=self.predict, size_hint=(1, .1), font_size="20sp", background_color=[0, 1, 0, 1])
        self.button2 = Button(
            text="Prediction Using MediaPipe Model", on_press=self.predict2, size_hint=(1, .1), font_size="20sp", background_color=[1, 0, 1, 1])
        self.prediction_label = Label(
            text="Prediction Results", size_hint=(1, .1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.button2)
        layout.add_widget(self.prediction_label)

        # Load tensorflow/keras model
        self.model = tf.keras.models.load_model("ASL_Model_Efficientnet_1.h5")
        # Load tensorflow/mediapipe model
        self.model2 = tf.keras.models.load_model("sign_classifier.h5")

        # Setup video Capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                            'del', 'nothing', 'space']

        self.class_names2 = [" A ", " B ", " C ", " E ", " G "]

        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        # Read frame from open-cv
        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

            ret, frame = self.capture.read()

            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing_styles.get_default_hand_landmarks_style(),
                                              mp_drawing_styles.get_default_hand_connections_style())

                    # Grab MIDDLE_FINGER_MCP coords
                    coords = tuple(np.multiply(
                        np.array(
                            (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y)), [640, 480]).astype(int))

                    # Get Decoded Hand box
                    cv2.rectangle(image,
                                  (coords[0] - 100, coords[1] - 100),
                                  (coords[0] + 100, coords[1] + 100),
                                  (245, 117, 16), 3)

            # Flip horizontall and convert image to texture
            buf = cv2.flip(image, 0).tostring()
            img_texture = Texture.create(
                size=(image.shape[1], image.shape[0]), colorfmt='bgr')
            img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.web_cam.texture = img_texture

    # Load image from file and conver to 200x200x
    def make_prediction(self, img, input_shape):

        image = cv2.resize(img, input_shape)
        img_reshaped = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2])
        pred = self.model.predict(img_reshaped)

        # if more than one output, take the max
        pred_class = self.class_names[pred.argmax()]
        pred_confidence = np.max(pred) * 100
        return pred_class, pred_confidence

    # Prediction function
    def predict(self, *args):

        ret, frame = self.capture.read()

        pred_class, pred_confidence = self.make_prediction(
            img=frame, input_shape=(200, 200))
        print(pred_class, pred_confidence)
        print(str(np.round(pred_confidence, 2)), " %")
        prob = str(np.round(pred_confidence, 2)) + " %"

        # Set verification text
        self.prediction_label.text = f"CLASS : {pred_class}       ||      CONFIDENCE : {prob}"

        # Log out details
        Logger.info(pred_class)
        Logger.info(pred_confidence)

        return pred_class, prob

# Prediction function 2
    def predict2(self, *args):

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.8) as hands:
            ret, frame = self.capture.read()

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Extract Pose landmarks
                    hand = hand_landmarks.landmark
                    hand_row = list(np.array(
                        [[landmark.x, landmark.y, landmark.z] for landmark in hand]).flatten())
                    # Make Detections
                    X = pd.DataFrame([hand_row])
                    pred = self.model2.predict(X)

                    sign_alphabet = self.class_names2[tf.argmax(
                        pred.flatten())]
                    sign_alphabet_prob = tf.reduce_max(pred).numpy()
                    sign_alphabet_prob = np.round(sign_alphabet_prob*100, 2)
                    print(sign_alphabet, sign_alphabet_prob)

        # Set verification text
        self.prediction_label.text = f"CLASS : {sign_alphabet}       ||      CONFIDENCE : {sign_alphabet_prob} %"

        # Log out details
        Logger.info(sign_alphabet)
        Logger.info(sign_alphabet_prob)

        return sign_alphabet, sign_alphabet_prob


if __name__ == '__main__':
    AmericanSignLanguageRecognitionApp().run()
