import face_recognition
from imutils import paths
import face_recognition
import pickle
import cv2



print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("./Face_expression/Known_Images"))
# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    get_name = imagePath.split("\\")
    temp = get_name.pop()
    temp = imagePath.split('_')
    name = get_name.pop()
    print(name)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='cnn')
	# compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb,boxes)
	# loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
		# encodings
        knownEncodings.append(encoding)
        knownNames.append(name)
    
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle","wb")
f.write(pickle.dumps(data))
f.close()