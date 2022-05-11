
from cProfile import label
from pathlib import Path
from tkinter import Image
from fer import FER
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os

def DetectFaceExpressions(foldername,name, size):

  captured_emotions_List = []
  emotion_Score_List = []
  captured_emotions_List.clear()
  emotion_Score_List.clear()
  path1 = 'Known_Images/' + foldername
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
  x = np.linspace(1,size, num= size) # total frames
  y = emotion_Score_List# time passed between the time each frame was taken

  plt.plot(x,y, 'o', ms=5)
  plt.title( name +' Emotions Over Time')
  plt.xlabel("Frames")
  plt.ylabel("Confidenct in Dominant Expression")
  for x in range(len(emotion_Score_List)):
    plt.text(x, emotion_Score_List[x], captured_emotions_List[x], fontsize = 8)    
  plt.show()



DetectFaceExpressions("teacher","Teachers", 17)
DetectFaceExpressions("Student_1","Student1", 16)
DetectFaceExpressions("Student_2", "Student2", 13)# nothing graphs
DetectFaceExpressions("Student_3", "Student3", 17)
DetectFaceExpressions("Student_4", "Student4", 13) # only one person
DetectFaceExpressions("Student_5", "Student5", 6) # three people
DetectFaceExpressions("Student_6", "Student6", 11) # 7 people
DetectFaceExpressions("Student_7", "Student7", 11) # everyone