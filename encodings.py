from itertools import count
from cv2 import imwrite
import face_recognition
import cv2
import os
import dlib
import pickle
import argparse
import numpy as np
from imutils import paths
import natsort
import multiprocessing as mp


"""ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image file')
ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
                help='path to weights file')
args = ap.parse_args()"""

imagePaths = natsort.natsorted(list(paths.list_images('./Face_expression/output')))
frame_path = 'Working_face_matches'
data = pickle.loads(open('encodings.pickle', "rb").read())



counter = 0




# The function that will apply to every image.
def imageFunc(imagePath):
        file_name = os.path.basename(imagePath)
        print(file_name)
        names = []
        known_encodings = data['encodings']
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='cnn')
        current_encoding = face_recognition.face_encodings(rgb, boxes)
        for encoding in current_encoding:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                first_match = matches.index(True)
                name = data["names"][first_match]
                names.append(name)
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(image, (left , top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                 0.75, (0, 255, 0), 2)
            # sh    ow the output image
            cv2.imwrite("./Face_expression/Working_face_matches/" + file_name, image)

        

# An output queue that will hold the results.
for (i, imagePath) in enumerate(imagePaths):
    if(counter == 300):
        break
    else:
        imageFunc(imagePath)