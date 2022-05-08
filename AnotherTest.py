

from fer import FER
import matplotlib.pyplot as plt 
import cv2

test_image_one = cv2.imread('elon2.jpg')
emo_detector = FER(mtcnn=True)
# Capture all the emotions on the image
captured_emotions = emo_detector.detect_emotions(test_image_one)
# Print all captured emotions with the image
print("Printing all captured emotions ", captured_emotions)
cv2.imshow('name', test_image_one)
cv2.waitKey()
#cv2.imshow('bob', test_image_one)
# Use the top Emotion() function to call for the dominant emotion in the image
dominant_emotion, emotion_score = emo_detector.top_emotion(test_image_one)
print(dominant_emotion, emotion_score)