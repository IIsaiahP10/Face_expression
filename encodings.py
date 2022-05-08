from itertools import count
import face_recognition
import cv2
import os
import dlib
import argparse




dataset_path = './Face_expression/dataset'
image_of_teach = face_recognition.load_image_file('./Face_expression/output/frame5400.jpg')
teach_face_encoding = face_recognition.face_encodings(image_of_teach)[0]

known_face_encodings = [
    teach_face_encoding

]

known_names = [
    "teacher"
]
count = 0
for images in os.listdir(dataset_path):
    print('Processing image {}'.format(count+1))
    count +=1
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        image = cv2.imread(dataset_path + '/' + images, 1)
        #rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2BGR)
        boxes = face_recognition.face_locations(image, model='cnn')
        encodings = face_recognition.face_encodings(image, boxes)
        for encoding in encodings:
                known_face_encodings.append(encoding)
    for encoding in known_face_encodings:
        print(encoding)

