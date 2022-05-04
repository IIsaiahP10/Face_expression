import face_recognition
import os
import numpy as np
from PIL import Image, ImageDraw
global STUDENT_COUNT
global COUNT
COUNT = 0
STUDENT_COUNT = 1
import os




def compareFaces(test_locations, test_encodings):
    global STUDENT_COUNT, COUNT
    pil_image = Image.fromarray(test_image)
    draw = ImageDraw.Draw(pil_image)
    for(top, right, bottom, left), test_encodings in zip(test_locations, test_encodings):
        matches = face_recognition.compare_faces(known_face_encoding, test_encodings)
        name = "Unknown person"
        face_distances = face_recognition.face_distance(known_face_encoding, test_encodings)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        else:
            known_face_encoding.append(test_encodings)
            known_face_names.append("Student" + str(STUDENT_COUNT))
            STUDENT_COUNT = STUDENT_COUNT + 1


        draw.rectangle(((left, top), (right,bottom)), outline=(0,0,0))

        #draw label

        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))
        draw.text((left+6, bottom - text_height - 5), name, fill=(255,255,255,255))

 
        

    del draw

    pil_image.save("image" + str(COUNT) + ".jpg")
    COUNT +=1

    


image_of_teach = face_recognition.load_image_file('./Face_expression/output/frame5400.jpg')
teach_face_encoding = face_recognition.face_encodings(image_of_teach)[0]


known_face_encoding = [
    teach_face_encoding]

known_face_names = [
    "teacher"
]


folder_dir = './Face_expression/output'
for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
        test_image = face_recognition.load_image_file(folder_dir + '/' + images)
        test_locations = face_recognition.face_locations(test_image)
        test_encodings = face_recognition.face_encodings(test_image, test_locations)
        compareFaces(test_locations, test_encodings)

