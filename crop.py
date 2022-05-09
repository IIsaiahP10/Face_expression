



import argparse

import time
import dlib
import cv2




# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image file')
ap.add_argument('-w', '--weights', default='./mmod_human_face_detector.dat',
                help='path to weights file')
args = ap.parse_args()

image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()
    

# initialize cnn based face detector with the weights
cnn_face_detector = dlib.cnn_face_detection_model_v1(args.weights)

start = time.time()

# apply face detection (cnn)
faces_cnn = cnn_face_detector(image, 1)

print("[INFO] Found {0} Faces." .format(len(faces_cnn)))

end = time.time()
print("CNN : ", format(end - start, '.2f'))


count = 0
# loop over detected faces
for face in faces_cnn:
	
	x = face.rect.left()
	y = face.rect.top()
	w = face.rect.right() - x
	h = face.rect.bottom() - y
	# draw box over face

	cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
	roi = image[y:y + h, x:x + w]
	print("[INFO] Object found. Saving locally.")
	cv2.imwrite(str(count) +  '_faces.jpg', roi)
	count +=1


img_height, img_width = image.shape[:2]
cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255), 2)

# display output image
cv2.imshow("face detection with dlib", image)
#cv2.waitKey()

# save output image 
cv2.imwrite("cnn_face_detection.png", image)

# close all windows
cv2.destroyAllWindows()