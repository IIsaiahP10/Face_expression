
from cProfile import label
from pathlib import Path
from tkinter import Image
from fer import FER
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os

captured_emotions_List = []
emotion_Score_List = []
path1 = 'Known_Images/teacher'
files = Path(path1).glob('*')
for file in files:
    fileImages = cv2.imread(str(file))
    # using the FER trained model to detect an emotion in the image
    emotion_detector = FER(mtcnn=True)
    # Capture all the emotions on the image
    captured_emotions = emotion_detector.detect_emotions(fileImages)
    # Use the top Emotion() function to call for the dominant emotion in the image
    dominant_Emotion, emotion_Score = emotion_detector.top_emotion(fileImages)
    captured_emotions_List.append(dominant_Emotion)
    emotion_Score_List.append(emotion_Score)
    # To see the images being analyzed for their emotion and score, uncomment the lines below
    #cv2.imshow(str(file), fileImages)
    #cv2.waitKey()
    #print(dominant_Emotion, emotion_Score)

x = np.linspace(1,17, num= 17) # total frames
y = emotion_Score_List# time passed between the time each frame was taken

plt.plot(x,y, 'o', ms=5)
plt.title( 'Teachers Emotions Over Time')
plt.xlabel("Frames")
plt.ylabel("Confidenct in Dominant Expression")
for x in range(len(emotion_Score_List)):
  plt.text(x, emotion_Score_List[x], captured_emotions_List[x], fontsize = 8)    
plt.show()
###Student 1
captured_emotions_List.clear()
emotion_Score_List.clear()
path1 = 'Known_Images/Student_1'
files = Path(path1).glob('*')
for file in files:
    fileImages = cv2.imread(str(file))
    # using the FER trained model to detect an emotion in the image
    emotion_detector = FER(mtcnn=True)
    # Capture all the emotions on the image
    captured_emotions = emotion_detector.detect_emotions(fileImages)
    # Use the top Emotion() function to call for the dominant emotion in the image
    dominant_Emotion, emotion_Score = emotion_detector.top_emotion(fileImages)
    captured_emotions_List.append(dominant_Emotion)
    emotion_Score_List.append(emotion_Score)
x = np.linspace(1,16, num= 16) # total frames
y = emotion_Score_List# time passed between the time each frame was taken

plt.plot(x,y, 'o', ms=5)
plt.title( 'Teachers Emotions Over Time')
plt.xlabel("Frames")
plt.ylabel("Confidenct in Dominant Expression")
for x in range(len(emotion_Score_List)):
  plt.text(x, emotion_Score_List[x], captured_emotions_List[x], fontsize = 8)    
plt.show()

